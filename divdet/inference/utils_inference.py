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
from itertools import zip_longest, repeat
import logging

import ogr
import numpy as np
from tqdm import tqdm
import skimage.io as sio
from skimage.transform import downscale_local_mean
from skimage.measure import regionprops, find_contours
from rasterio.windows import Window


def iter_grouper(iterable, n, fillvalue=None):
    "Itertool recipe to collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n

    return zip_longest(*args, fillvalue=fillvalue)


def calculate_region_grad(image, masked_pix):
    """Calculate the vertical and horizontal gradient using numpy's gradient.

    Parameters
    ----------
    image: array-like
        Image pixels.
    masked_pix: array-like
        Integer mask for pixels to include when returning mean gradient.
        For example, pass the binary crater mask to calculate only the gradient
        for pixels within the crater.

    Returns
    -------
    mean_h: float
        Mean horizontal gradient of included pixels. Positive is to the right.
    mean_v: float
        Mean vertical gradient of included pixels. Positive is downward.
    """
    # TODO: could improve function to take a list of good_inds and avoid
    # recomputing gradient repeatedly

    if masked_pix.dtype != np.bool:
        raise ValueError('`masked_inds` must be of type bool')
    if masked_pix.shape[:2] != image.shape[:2]:
        raise ValueError('Height and width of `masked_inds` must match `image`')

    mean_h = np.ma.masked_array(np.gradient(image, axis=0),
                                mask=masked_pix).mean()
    mean_v = np.ma.masked_array(np.gradient(image, axis=1),
                                mask=masked_pix).mean()
    return mean_h, mean_v


def calculate_shape_props(mask):
    """Calculate simple geometric metrics for a binary mask.

    Parameters
    ----------
    mask: numpy.ndarray
        2 dimensional integer image containing a single object blob (e.g., a
        mask for one crater).
    """

    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise RuntimeError('`mask` must be 2D.')
    if mask.dtype != np.int:
        raise ValueError('`mask` must be of type int')

    props = regionprops(mask)
    return props[0]  # Assume only one shape is in the mask


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
    if not (isinstance(slice_size[0], int) and isinstance(slice_size[1], int)):
        raise ValueError('Slice sizes aren\'t integers.')
    if not (isinstance(min_window_overlap[0], int) and isinstance(min_window_overlap[1], int)):
        raise ValueError('Window overlap sizes aren\'t integers.')

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


def windowed_reads_rasterio(tif, slice_coords):
    """Make a series of windowed reads

    Parameters
    ----------
    tif: open rasterio image
    slice_coords: list
        List of (x, y, width, height) rows in pixel coords to make windowed
        reads.
    """

    # Loop through all windows and save if they don't exist
    windows = []
    for (x_pt, y_pt, width, height) in slice_coords:
        # Rasterio works in `xy` coords, not `ij`
        windows.append(tif.read(1, window=Window(y_pt, x_pt, width, height)))

    return windows


def yield_windowed_reads_rasterio(tif, slice_coords):
    """Make a series of windowed reads via generator

    Parameters
    ----------
    tif: open rasterio image
    slice_coords: list
        List of (x, y, width, height) rows in pixel coords to make windowed
        reads.
    """

    # Loop through all windows and save if they don't exist
    for (x_pt, y_pt, width, height) in slice_coords:
        # Rasterio works in `xy` coords, not `ij`
        yield tif.read(1, window=Window(y_pt, x_pt, width, height))


def _cut_window(arr, row, col, width, height):
    """Cut a slice out of an array given coords and width/height"""

    row, col = int(np.round(row)), int(np.round(col))
    height, width = int(np.round(height)), int(np.round(width))

    # Slice array
    sub_arr = arr[row:row + height, col:col + width, ...]

    if sub_arr.shape[:2] != (height, width):
        sub_arr = np.pad(sub_arr, (height, width))

    return sub_arr


def windowed_reads_numpy(arr, slice_coords):
    """Make a series of windowed reads

    Parameters
    ----------
    arr: np.ndarray
        Image data
    slice_coords: list
        List of (x, y, width, height) rows in pixel coords to make windowed
        reads.
    """
    windows = []
    # Store all windows
    for (row, col, height, width) in slice_coords:
        sub_arr = _cut_window(arr, row, col, width, height)
        windows.append(dict(image_data=sub_arr, row=row, col=col,
                            height=height, width=width))
        # Create a contiguous array (necessary for b64 encoding)
        #windows.append(dict(image_data=np.ascontiguousarray(sub_arr),
        #                    row=row, col=col, height=height, width=width))

    return windows


def yield_windowed_reads_numpy(arr, slice_coords):
    """Make a series of windowed reads via generator

    Parameters
    ----------
    arr: np.ndarray
        Image data
    slice_coords: list
        List of (x, y, width, height) rows in pixel coords to make windowed
        reads.
    """

    # Yield all windows
    for (row, col, height, width) in slice_coords:
        sub_arr  = _cut_window(arr, row, col, width, height)

        # Create a contiguous array (necessary for b64 encoding)
        #yield dict(image_data=np.ascontiguousarray(sub_arr), row=row, col=col,
        yield dict(image_data=sub_arr, row=row, col=col, height=height, width=width)


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

    Notes
    -----
    Adapted from Malisiewicz et al.

    Consider using TensorFlows built in non-max suppression functionality.
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


def poly_non_max_suppression(polys, confidences, overlap_thresh=0.2,
                             mp_processes=None, mp_chunksize=None):
    """Non-maximum suppression on OGR/GDAL polygons to find duplicates.

    Parameters
    ----------
    polys: list
        List of OGR/GDAL polygons.
    confidences: np.ndarray
        Confidences on interval [0, 1] corresponding to prediction score of polygons in `polys`.
    overlap_thresh: float
        Intersection-over-union threshold to remove polygons.
    mp_processes: None or int
        Number of multiprocessing processes to start. Generally, want this to
        equal the number of cores you have.
    mp_chunksize: None or int
        Approximate number of chunks to divide the iterable into.

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
    if not mp_chunksize:
        mp_chunksize = 1

    # Check for empty polygons
    bad_poly_inds = [ind for ind in range(len(polys))
                     if polys[ind].IsEmpty()]
    if bad_poly_inds:
        logging.warning('Found {len(bad_poly_inds)} indices with empty polygons')
        for bad_ind in bad_poly_inds:
            del polys[bad_ind]
        confidences = confidences[np.invert(np.array(bad_poly_inds))]

    picks = []  # List with poly indices to keep

    # Order candidate confidences in ascending order
    candidate_inds = np.argsort(confidences)

    while candidate_inds.size:
        # Append best current score to picks list and delete it from candidates
        picks.append(candidate_inds[-1])
        candidate_inds = np.delete(candidate_inds, -1)
        if candidate_inds.size == 0:
            break

        # Get the IOU between the kept polygon and all remaining polygons

        # Pass geometries directly
        #starmap_list = [(polys[picks[-1]].Clone(), polys[ind].Clone(), overlap_thresh)
        #                for ind in candidate_inds]

        # Pass well known binary of geometries
        starmap_list = [(polys[picks[-1]].ExportToWkb(), polys[ind].ExportToWkb(), overlap_thresh)
                        for ind in candidate_inds]

        with multiprocessing.Pool(processes=mp_processes) as pool:
            overlap_iou = pool.starmap(poly_iou, starmap_list, mp_chunksize)

        # Use single threading instead of  multiprocessing
        #overlap_iou = []
        #for ind in candidate_inds:
        #    overlap_iou.append(poly_iou(polys[picks[-1]], polys[ind], overlap_thresh))

        # Try with pass in poly_iou
        # Tips/gotchas for gdal polys
        # Higher confidence
        # Explicit deletes
        # Bbox non-max-suppression


        # Remove indices of polygons that overlapped at or above the threshold
        #     by only keeping indices where `overlap_thresh` was not met
        candidate_inds = candidate_inds[np.invert(overlap_iou)]

    return picks


def poly_iou(poly1, poly2, thresh=None):
    """Compute intersection-over-union for two GDAL/OGR geometries.

    Parameters
    ----------
    poly1:
        First polygon used in IOU calc.
    poly2:
        Second polygon used in IOU calc.
    thresh: float or None
        If not provided (default), returns the float IOU for the two polygons.
        If provided, return True if the IOU met this threshold. Otherwise,
        False.

    Returns
    -------
    IOU: float or bool
        Return the IOU value if `thresh` is None, otherwise boolean if the
        threshold value was met.
    """

    poly1 = ogr.CreateGeometryFromWkb(poly1)
    poly2 = ogr.CreateGeometryFromWkb(poly2)

    intersection = poly1.Intersection(poly2)
    intersection_area = intersection.Area()

    if intersection == 0:
        return False

    union = poly1.Union(poly2)
    union_area = union.Area()

    # If threshold was provided, return if IOU met the threshold
    if thresh is not None:
        return (intersection_area / union_area) >= thresh

    return intersection_area / union_area


def convert_mask_to_polygon(mask, xy_offset=(0, 0), simplify_tol=0.5,
                            conf_thresh=0.5):
    """Convert a [0, 1] confidence mask to a GDAL polygon"""

    # Create padded mask
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.float)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, conf_thresh)  # Gives n (row, col) coordinates

    gdal_ring = ogr.Geometry(ogr.wkbLinearRing)

    if len(contours) > 1:
        raise RuntimeError('Found multiple mask contours for 1 crater.')

    # Remove the padding and transition from (row, col) to (x, y)
    contour = np.fliplr(contours[0] - 1)

    # Convert predicted mask to polygon
    for verts in contour:
        x = verts[0] + xy_offset[0]
        y = verts[1] + xy_offset[1]
        gdal_ring.AddPoint_2D(x, y)
    x0 = contour[0, 0] + xy_offset[0]
    y0 = contour[0, 1] + xy_offset[1]
    gdal_ring.AddPoint_2D(x0, y0)  # Add 0th point to close the polygon

    # Create polygon
    gdal_poly = ogr.Geometry(ogr.wkbPolygon)
    gdal_poly.AddGeometry(gdal_ring)

    # Attempt simplification
    gdal_poly_simp = gdal_poly.Clone()
    gdal_poly_simp.SimplifyPreserveTopology(simplify_tol)

    # Return original polygon if there is an issue during simplification
    if gdal_poly_simp.IsEmpty():
        logging.warning('Simplified polygon is empty. Using original polygon.')
        return gdal_poly

    return gdal_poly_simp


def geospatial_polygon_transform(poly, transform):
    """Convert a polygon from pixel coords to geospatial coords

    Parameters
    ----------
    poly: OGR Geometry
        Polygon in original coordinates (e.g., pixels)
    transform: rasterio dataset transform
        A mapping from one coordinate frame to another (e.g., pixels to lat/lon)

    Returns
    -------
    poly_transformed: OGR Geometry
        Polygon with all points transformed to the new coordinates
    """

    # Convert predicted mask to polygon
    gdal_ring = ogr.Geometry(ogr.wkbLinearRing)

    # Get single geometry from the poly object
    geom = poly.GetGeometryRef(0)
    for pi in range(geom.GetPointCount()):
        pt = transform * geom.GetPoint(pi)[:2]  # Only take first 2 coords in case 3 exist
        gdal_ring.AddPoint(*pt)

    # Create polygon
    gdal_poly = ogr.Geometry(ogr.wkbPolygon)
    gdal_poly.AddGeometry(gdal_ring)

    gdal_poly.FlattenTo2D()

    return gdal_poly
