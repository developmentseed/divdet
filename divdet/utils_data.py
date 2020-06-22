"""
Utilities for processing data (mostly at training time)

utils_data.py
"""

#import os.path as op
import xml.etree.ElementTree as ET

import numpy as np
from scipy import signal, sparse
from skimage import io as sio
from skimage.transform import downscale_local_mean
from skimage.color import gray2rgb
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm


def parse_xml_bbs(filepath):
    """Load and convert XML from CVAT into list of bounding boxes

    OpenCV's Computer Vision Annotation Tool is here:
    https://github.com/opencv/cvat

    Parameters
    ----------
    filepath: str
        Path to XML file on disk.

    Returns
    -------
    all_image_annots: list
        List of bounding box tuples (X-TL, Y-TL, X-BR, Y-BR) for all images in
        the XML file. This format is reversed from numpy coordinate ordering.
    """

    # Error checking
    root = ET.parse(filepath).getroot()
    children = [child.tag for child in root]
    if 'meta' not in children or 'image' not in children:
        raise ValueError('Problem reading XML, missing features')

    print("Reading image annotations for task: {}\n".format(
        root.find('meta').find('task').find('name').text))
    image_entries = root.findall('image')
    all_image_annots = []

    for ii, image in enumerate(image_entries):
        print('\t{}. Loading bboxes for {}'.format(ii, image.get('name')))

        # Loop through all bboxes in image and save them as list of tuples
        img_box_list = []
        for bb in image.findall('box'):
            coords = [round(float(bb.get(coord_key))) for coord_key in
                      ['xtl', 'ytl', 'xbr', 'ybr']]
            try:
                label = bb.find('attribute').text
            except AttributeError:
                label = 0

            img_box_list.append([tuple(coords), label])

        all_image_annots.append(img_box_list)

    return all_image_annots


def load_dataset(fpaths_img, fpaths_xml):
    """Load a list of images and corresponding xml from disk

    Parameters
    ----------
    fpaths_img: list of str
        Filepaths to images that will be loaded
    fpaths_xml: list of str
        Corresponding filepaths to xml containing object bboxes

    Returns
    -------
    img_arr: numpy.ndarray
        Images loaded into memory
    xml_list: list of list of tuple
        Object bounding boxes loaded into memory
    """

    img_list, xml_list = [], []
    for fpath_img, fpath_xml in zip(fpaths_img, fpaths_xml):
        img_list.append(sio.imread(fpath_img))
        xml_list.append(parse_xml_bbs(fpath_xml))

    return img_list, xml_list


def _get_all_valid_corners(img_arr, crop_size, l_thresh, corner_thresh):
    """Get all valid corners for random cropping"""

    valid_pix = img_arr >= l_thresh
    kernel = np.ones((crop_size, crop_size))

    conv = signal.correlate2d(valid_pix, kernel, mode='valid')

    return conv > (corner_thresh * crop_size ** 2)


def _get_random_valid_corners(img_arr, crop_size, n_crops, l_thresh,
                              corner_thresh):
    """Get all valid corners for random cropping deterministically"""

    valid_pix = img_arr >= l_thresh
    kernel = np.ones((crop_size, crop_size))

    conv = signal.correlate2d(valid_pix, kernel, mode='valid')

    vc = conv > (corner_thresh * crop_size ** 2)

    vc_sparse = sparse.csr_matrix(vc)
    samples = np.random.choice(vc_sparse.nnz, n_crops, replace=False)

    # Sample some corner indices
    corner_inds = []
    rows, cols = vc_sparse.nonzero()
    for samp in samples:
        corner_inds.append([rows[samp], cols[samp]])

    return corner_inds


def _get_valid_corners_iteratively(img_arr, crop_size, n_crops, l_thresh,
                                   corner_thresh):
    """Get all valid corners for random cropping"""

    vcs = []  # Valid corner pixels
    crop_area = crop_size ** 2

    # Invalidate areas where there aren't enough pixels to make a crop
    valid_pix = np.ones_like(img_arr, dtype=bool)
    valid_pix[:, -crop_size + 1:] = False
    valid_pix[-crop_size + 1:, :] = False

    # Get all possible corner pixel positions
    valid_pix_choices = np.stack(np.where(valid_pix), axis=-1)
    valid_pix_inds = np.arange(len(valid_pix_choices)).tolist()

    while len(vcs) < n_crops:
        if not valid_pix_inds:
            raise RuntimeError('Not enough corner inds for desired `n_crops`')

        samp_ind = valid_pix_inds.pop(np.random.choice(len(valid_pix_inds)))

        x_val, y_val = valid_pix_choices[samp_ind, :]

        # Get proportion of valid pixels in window
        window = img_arr[x_val:x_val + crop_size, y_val:y_val + crop_size]
        wind_sum = np.sum(window >= l_thresh)
        prop_valid = wind_sum / crop_area

        # Make sure the number of valid pixels meets the threshold
        if np.sum(prop_valid >= corner_thresh):
            vcs.append([x_val, y_val])

    return vcs


def slice_img(img_arr, bboxes, crop_size=512, n_crops=50, level_thresh=1,
              valid_pix_thresh=0.5, annot_overlap_thresh=0.25,
              annot_size_thresh=0.0001):
    """Randomly crop an image into smaller images

    Parameters
    ----------
    img_arr: np.ndarray
        Large numpy array representing image to be sliced up
    bboxes: list of tuple
        Bounding box annotations for the entire image. Should be fed from the
        `parse_xml_bbs` function.
    crop_size: int
        Size of the output cropped images. Default: 512.
    n_crops: int
        Number of random crops to make. Default: 50.
    level_thresh: int
        Minimum pixel level/intensity for that pixel to be designated as valid.
        This number should be higher than whatever value is assigned to
        "no-data" pixels. Default: 1
    valid_pix_thresh: float
        Used to control the proportion of pixels allowed to be invalid (e.g.,
        zero-padded). When a sliding window of `crop_size` is
        cross-correlated with a region, this defines the proportion of valid
        pixels that must be present for that region to remain a valid
        potential crop. Default: 0.5
    annot_overlap_thresh: float
        Used to control which bounding box annotations to include. This sets
        a lower bound on the proportion of the annotation area that must
        intersect with the crop image bounding box -- basically, a filter to
        exclude bounding boxes that poorly overlap the image crop.
        Default: 0.25
    annot_size_thresh: float
        Used to control which bounding box annotations to include. This sets
        a lower bound on the minimum relative size of the intersecting
        annotation area compared to the entire image -- basically, a filter to
        exclude bounding boxes that are tiny relative to the image crop.
        Default: 0.0001

    Returns
    -------
    crops: list
        List numpy array containing HWC cropped images.
    annots: list
        List of bounding boxes associated with each image.
    """

    # Get valid corners and sample from them
    # Deterministic corner fetching is very slow
    #corner_inds = _get_valid_corners(img_arr, crop_size, n_crops,
    #                                 level_thresh, valid_pix_thresh)

    # Use randomized corner sampling
    corner_inds = _get_valid_corners_iteratively(img_arr, crop_size, n_crops,
                                                 level_thresh,
                                                 valid_pix_thresh)

    # Crop images and bounding boxes
    crops = []
    annots = []
    min_annot_size = annot_size_thresh * (crop_size ** 2)
    for ci in tqdm(corner_inds, 'Valid corner indicies:'):
        # Get cropped image data
        temp_crop = img_arr[ci[0]:ci[0] + crop_size,
                            ci[1]:ci[1] + crop_size]
        if len(temp_crop.shape) == 2:
            temp_crop = temp_crop.reshape(temp_crop.shape[0],
                                          temp_crop.shape[1], 1)
        img_bb = ia.BoundingBox(ci[1], ci[0],
                                ci[1] + crop_size, ci[0] + crop_size)

        # Find valid bounding box annotations for image
        temp_bboxes = []
        for bbox in bboxes:
            bbox_coords = bbox[0]
            try:
                obj_bb = ia.BoundingBox(bbox_coords[0], bbox_coords[1],
                                        bbox_coords[2], bbox_coords[3],
                                        label=bbox[1])
            except AssertionError:
                # Catch errors where bbox is invalid
                continue

            try:
                intersection = img_bb.intersection(obj_bb).area
            except AttributeError:
                # Catch errors for 0 overlap
                intersection = 0

            if (intersection >= (obj_bb.area * annot_overlap_thresh) and
                    (obj_bb.area >= min_annot_size)):
                obj_bb = obj_bb.shift(bottom=ci[0], right=ci[1])
                temp_bboxes.append(obj_bb)

        crops.append(temp_crop)
        annots.append(temp_bboxes)

    return crops, annots


def augment_img(image, annots, n_augs=10, seed=None):
    """Augment a single img/bbox pair with imgaug

    Parameters
    ----------
    image: np.ndarray
        uint8 array containing image information. Should be BHWC or HWC.
    annots: list
        Contains bbox integers and annotation information.
    n_augs: int
        Number of augmented images to generate. Default: 10.
    seed: int
        Seed for imgaug sequential randomization.

    Returns
    -------
    image_augs: np.ndarray
        Augmented images stored in one array
    annot_augs: list
        Bounding boxes for augmented images
    annotated_imgs: np.ndarray
        Images with bounding boxes draw on them (for visualization)
    """

    if not image.dtype == np.uint8:
        raise ValueError('`image` must be of type uint8')
    if len(image.shape) == 3:
        image = image[np.newaxis, ...]
    if not len(image.shape) == 4:
        raise ValueError('`image` must be 3 or 4 dimensional')

    if seed is not None:
        ia.seed(seed)

    ia_bbs = ia.BoundingBoxesOnImage(annots, shape=image.squeeze().shape)

    seq = iaa.Sequential([
        iaa.Multiply((0.8, 1.2)), # change brightness, doesn't affect BBs
        iaa.Fliplr(0.5), # horizontal flip
        iaa.Flipud(0.5), # vertically flip
        iaa.ContrastNormalization((0.5, 2.0)),
        #iaa.Affine(scale=(1., 1.5))#, shear=(-15, 15))
        ])

    image_augs = np.zeros((n_augs, image.shape[1], image.shape[2],
                           image.shape[3]), dtype=np.uint8)
    annotated_image_augs = np.tile(image_augs, (1, 1, 1, 3))
    annot_augs = []

    for batch_idx in range(n_augs):
        # IMPORTANT: Call this once PER BATCH, otherwise you will always get
        # exactly the same augmentations for every batch!
        seq_det = seq.to_deterministic()

        image_augs[batch_idx] = seq_det.augment_images(image)[0]
        annot_aug = []
        if ia_bbs.bounding_boxes:
            annot_aug = seq_det.augment_bounding_boxes([ia_bbs])[0]

            # image with BBs after augmentation
            annot_aug = annot_aug.remove_out_of_image().cut_out_of_image()
            annotated_image_augs[batch_idx, :, :, :] = annot_aug.draw_on_image(
                gray2rgb(image_augs[batch_idx].squeeze()), thickness=2)
        else:
            annotated_image_augs[batch_idx] = image_augs[batch_idx]

        # Store augmentation
        annot_augs.append(annot_aug)

    return image_augs, annot_augs, annotated_image_augs


def format_image_bboxes(bboxes_on_image, class_dict, crop_size=512):
    """Convert ImageAug BoundingBoxOnImage object to YOLO fmt labels

    Parameters
    ----------
    bboxes_on_image: imgaug.imgaug.BoundingBoxexOnImage
        ImgAug object containing all bboxes
    class_dict: dict
        Dictionary containing class labels as keys and class numbers as objects

    Returns
    -------
    formatted_bboxes: list
        List of strings formatted in YOLO format for writing to text file.
    """

    formatted_bboxes = []

    # Ensure that some bounding boxes exist
    if bboxes_on_image:
        for bbox in bboxes_on_image.bounding_boxes:
            coords = [(bbox.x1 + bbox.width / 2.) / crop_size,
                      (bbox.y1 + bbox.height /2.) / crop_size,
                      bbox.width / crop_size,
                      bbox.height / crop_size]
            for coord_i, coord in enumerate(coords):
                if coord <= 0:
                    coords[coord_i] = 1e-8
                if coord >= 1:
                    coords[coord_i] = 1 - 1e-8

            line = '{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n'.format(
                cls=bbox.label, x=coords[0], y=coords[1], w=coords[2],
                h=coords[3])
            formatted_bboxes.append(line)

    return formatted_bboxes


def multi_scale_slice(img_arr, bboxes, n_crops_per_scale=10,
                      downscales=[2, 4, 8], slice_img_params=None):
    """Slice image at multiple scales

    n_crops_per_scale: int or list of int
        Number of crops per downscale value
    downscales: list of int
        Downscaling factor to modify size of image before cropping.
    slic_img_params: dict
        Arguments for `slice_img` function to be passed through.
    """

    assert isinstance(n_crops_per_scale, (int, list))
    if isinstance(n_crops_per_scale, int):
        n_crops_per_scale = [n_crops_per_scale] * len(downscales)
    if slice_img_params is None:
        slice_img_params = {}


    # Crop images and bounding boxes
    crops = []
    annots = []

    for n_crops, downscale in zip(n_crops_per_scale, downscales):

        scaled_img = downscale_local_mean(img_arr, (downscale, downscale)).astype(img_arr.dtype)
        scaled_bboxes = [[tuple([int(np.round(val / downscale))
                                 for val in bbox[0]]), bbox[1]]
                         for bbox in bboxes]

        crop_batch, annot_batch = slice_img(scaled_img, scaled_bboxes,
                                            n_crops=n_crops,
                                            **slice_img_params)

        crops.extend(crop_batch)
        annots.extend(annot_batch)

    return crops, annots
