"""
Tests for inference utilities.
"""

import numpy as np
from osgeo import ogr
from divdet.inference.utils_inference import (poly_non_max_suppression, poly_iou,
                                              get_slice_bounds, yield_windowed_reads_rasterio,
                                              windowed_reads_rasterio)


from numpy.testing import assert_array_equal, assert_almost_equal

wkt = "POINT (1120351.5712494177 741921.4223245403)"
point = ogr.CreateGeometryFromWkt(wkt)

# Create Rectangles
rect1_wkt = 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'  # rect from (0, 0) to (1, 1)
rect2_wkt = 'POLYGON ((0 0, 0.5 0, 0.5 0.5, 0 0.5, 0 0))' # rect from (0, 0) to (0.5, 0.5)
rect3_wkt = 'POLYGON ((0.1 0.1, 1.1 0.1, 1.1 0.1, 0.1 1.1, 0.1 0.1))' # rect from (0.1, 0.1) to (1.1, 1.1)
rect4_wkt = 'POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))' # rect from (1, 1) to (2, 2)

rect1 = ogr.CreateGeometryFromWkt(rect1_wkt)
rect2 = ogr.CreateGeometryFromWkt(rect2_wkt)
rect3 = ogr.CreateGeometryFromWkt(rect3_wkt)
rect4 = ogr.CreateGeometryFromWkt(rect4_wkt)


class TestPolygons:
    """Various tests for verifying crater polygons"""

    def test_poly_iou(self):
        """Test intersection over union calculations"""

        assert_almost_equal(poly_iou(rect1, rect2), 0.25)
        assert poly_iou(rect1, rect2, thresh=0.24)
        assert not poly_iou(rect1, rect2, thresh=0.26)


    def test_poly_non_max_suppression(self):
        """Test that repeated detections are removed"""
        picks = poly_non_max_suppression([rect1, rect2], [0.51, 0.5])
        assert_array_equal(picks, [0, 1])

        picks = poly_non_max_suppression([rect1, rect3], [0.5, 0.6])
        assert_array_equal(picks, [1])

        picks = poly_non_max_suppression([rect1, rect3], [0.6, 0.5])
        assert_array_equal(picks, [0])

        picks = poly_non_max_suppression([rect1, rect3], [0.5, 0.6])
        assert_array_equal(picks, [1])

        picks = poly_non_max_suppression([rect1, rect2, rect3, rect4], [0.9, 1., 0.8, 0.1])
        assert_array_equal(picks, [1, 0, 3])

        picks = poly_non_max_suppression([rect1, rect1, rect1], [0.9, 0.99, 1])
        assert_array_equal(picks, [2])


class TestSlicing:
    """Tests for cutting an image into smaller slices prior to ML inference"""

    def test_slice_coords(self):
        image_size = (150, 150)
        slice_size = (100, 100)
        min_window_overlap = (50, 50)

        # Test optimal slices where overlap fits evenly into image
        slices = get_slice_bounds(image_size, slice_size, min_window_overlap)
        assert_array_equal(slices, np.array([[0, 0, 100, 100],
                                             [0, 50, 100, 100],
                                             [50, 0, 100, 100],
                                             [50, 50, 100, 100]]))

        # Test case where slices overlap doesn't exactly match
        image_size = (120, 120)
        slices = get_slice_bounds(image_size, slice_size, min_window_overlap)
        assert_array_equal(slices, np.array([[0, 0, 100, 100],
                                             [0, 20, 100, 100],
                                             [20, 0, 100, 100],
                                             [20, 20, 100, 100]]))
