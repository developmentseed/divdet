"""
Tests for inference utilities.
"""

from osgeo import ogr
from divdet.inference.utils_inference import poly_non_max_suppression, poly_iou

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
