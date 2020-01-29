"""
Utilities for running inference

@author: Development Seed
"""
import os
from os import path as op
import multiprocessing
from functools import partial

import subprocess
import warnings
from itertools import zip_longest

import numpy as np
from tqdm import tqdm
import skimage.io as sio
from skimage.transform import downscale_local_mean
from rasterio.windows import Window
from geojson import Feature, Polygon


def iter_grouper(iterable, n, fillvalue=None):
    "Itertool recipe to collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n

    return zip_longest(*args, fillvalue=fillvalue)


def get_slice_bounds(image_size, slice_size=(1024, 1024),
                     min_window_overlap=(128, 128)):
    """Get list of overlapping window bounds for a larger image.

    Parameters
    ----------
    image_size: len-2 iterable
        Size of the original image for windowing (height, width).
    slice_size: 2-tuple of int
        Size of the windowed reads to make.
    min_window_overlap: 2-tuple of int
        Minimum pixel overlap between images.

    Returns
    -------
    slice_coords: list of tuple
        Pixel coordinates (row, col, window_size_x, window_size_y)
        useful for making windowed reads into a larger image.
    """
    assert isinstance(slice_size[0], int)
    assert isinstance(slice_size[1], int)
    assert isinstance(min_window_overlap[0], int)
    assert isinstance(min_window_overlap[1], int)

    ideal_intervals = [win_dim - overlap for win_dim, overlap
                       in zip(slice_size, min_window_overlap)]
    n_windows = [np.int(np.ceil((img_dim - win_size) / interval))
                 for img_dim, win_size, interval
                 in zip(image_size, slice_size, ideal_intervals)]

    y_vals = np.linspace(0, image_size[0] - slice_size[0],
                         n_windows[0] + 1, endpoint=True)
    x_vals = np.linspace(0, image_size[1] - slice_size[1],
                         n_windows[1] + 1, endpoint=True)

    # Create meshgrid to get all points w/out for loop
    rows, cols = np.meshgrid(y_vals, x_vals, indexing='ij')  # Use `ij` indexing
    slice_coords = [(row_val, col_val, slice_size[0], slice_size[1])
                     for row_val, col_val in zip(rows.ravel(), cols.ravel())]

    return slice_coords


def yield_windowed_reads(tif, slice_coords):
    """Make a series of windowed reads via generator

    Parameters
    ----------
    tif: open rasterio image
    slice_coords: list
        List of (x, y, width, height) rows in pixel coords to make windowed
        reads.
    """

    # Loop through all windows and save if they don't exist
    for (x_pt, y_pt, width, height) in tqdm(slice_coords):
        # Rasterio works in `xy` coords, not `ij`
        yield tif.read(1, window=Window(y_pt, x_pt, width, height))


def non_max_suppression(bboxes, overlap_thresh=0.4):
    """Run non-maximum suppression to avoid duplicate detections.

    Sorts by confidence value

    bboxes: ndarray
            Numpy array where each row is <x1> <y1> <x2> <y2> <confidence>

    overlap_thresh: float
            Ratio of area overlap to remove bboxes.

    Returns
    -------
    picks: list of int
            Good indices to choose from bboxes array

    Note: Adapted from Malisiewicz et al.
    """

    # Error check for no inputs
    if len(bboxes) == 0:
        return []

    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype(np.float)

    picks = []

    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]
    scores = bboxes[:, 4]

    # Compute the area of the bounding bboxes and sort by the
    #   confidence score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs):
        # Get the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        picks.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list where overlap exceeds threshold
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlap_thresh)[0])))

    # return indices for bounding bboxes that were picked
    return picks


def poly_non_max_suppression(polys, confidences, overlap_thresh=0.4, multi_proc_chunksize=None):
    """Non-maximum suppression on OGR/GDAL polygons to find duplicates.

    Parameters
    ----------
    polys: list
        List of OGR/GDAL polygons.
    confidences: np.ndarray
        Confidences on interval [0, 1] corresponding to prediction score of polygons in `polys`.
    overlap_thresh: float
        Intersection-over-union threshold to remove polygons.

    Returns
    -------
    picks: list of int
        Good indices to keep in polys list.
    """

    # Error check for no inputs
    if len(polys) == 0:
        return []
    if not len(polys) == len(confidences):
        raise ValueError("`polys` and `confidences` do not have the same length.")
    if not 0 <= overlap_thresh <= 1.:
        raise ValueError("`overlap_thres` must be on interval [0, 1].")

    picks = []
    conf_inds = np.argsort(confidences) # Order confidences in ascending order

    while len(conf_inds):

        # Append best current score to picks list
        picks.append(conf_inds[-1])

        # Find all polygons that overlap with most confident polygon
        iou_partial_func = partial(poly_iou, poly1=polys[-1], thresh=overlap_thresh)
        overlap_list = multiprocessing.imap(iou_partial_func, [poly[ind] for ind in conf_inds], multi_proc_chunksize)

        # Remove overlapping polygons
        overlap_arr = np.array(overlap_list)
        conf_inds = np.delete(conf_inds, np.concatenate(
            ([-1], np.where(overlap_arr)[0])))

    return picks


def poly_iou(poly1, poly2, thresh=None):
    """Compute intersection-over-union for two GDAL/OGR geometries."""

    intersection = poly1.Intersection(poly2)
    union = poly1.Union(poly2)

    # If threshold was provided, return if IOU met the threshold
    if thresh is not None:
        return (intersection / union) >= thresh

    return intersection / union


def run_darknet_test(dir_darknet, fpath_data_cfg, fpath_model_cfg,
                     fpath_weights, fpath_images, thresh=0.4):
    """Execute prediction using darknet.

    Parameters
    ----------
    dir_darknet: str
        Directory of darknet codebase
    fpath_data_cfg: str
        Data configuration file for YOLO
    fpath_model_cfg: str
        Model configuration file for YOLO
    fpath_weights: str
        Model weights for YOLO
    fpath_images: str
        Text file containing file paths to all images for inference
    thresh: float
        Confidence threshold to include a prediction
    """

    p_files = subprocess.Popen(['cat', '--squeeze-blank', fpath_images], stdout=subprocess.PIPE,
                               shell=False)
    p_darknet = subprocess.Popen([op.join(dir_darknet, 'darknet'), 'detector', 'test', fpath_data_cfg, fpath_model_cfg,
         fpath_weights, '-thresh', str(thresh), '-save_labels', '-dont_show',
         '-ext_output'], stdin=p_files.stdout, stdout=subprocess.PIPE, shell=False, cwd=dir_darknet)

    p_files.stdout.close()
    return p_darknet.communicate()[0]

    '''
    # Compile command
    run(['cd', dir_darknet])
    run([op.join(dir_darknet, 'darknet'), 'detector', 'test', fpath_data_cfg, fpath_model_cfg,
         fpath_weights, '-thresh', str(thresh), '-save_labels', '-dont_show',
         '-ext_output <', all_files])
    '''

    '''
    # This method is a security risk if we start taking external output
    cmd = ('./darknet detector test {} {} {} -thresh {} -save_labels -dont_show -ext_output < {}'.format(
           fpath_data_cfg, fpath_model_cfg, fpath_weights, thresh, fpath_images))
    subprocess.run(cmd, cwd=dir_darknet, shell=True)
    '''



def convert_to_geojson(bboxes, properties):
    """Write a set of bboxes to a geojson string.

    Parameters
    ----------
    bboxes: ndarray
        Bbox coordinates where each row is <x1> <y1> <x2> <y2>. Points
        correspond to LL and UR coordinates.
    properties: list
        List of property dicts (matching length of bboxes array) to be written
        to `properties` key in each geojson Feature.

    Returns
    -------
    geojson_features: list of str
        List of features ready to be converted to FeatureCollection and
    	saved to disk with geojson.dump
    """
    if not len(bboxes) == len(properties):
        raise ValueError('Length of bboxes and properties must match')

    geojson_features = []
    for (x1, y1, x2, y2), prop_dict in zip(bboxes, properties):
        geojson_features.append(Feature(properties=prop_dict,
                                        geometry=Polygon([[[x1, y1],
                                                           [x2, y1],
                                                           [x2, y2],
                                                           [x1, y2],
                                                           [x1, y1]]])))

    return geojson_features
