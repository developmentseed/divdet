"""
Test surface feature component of divdet
"""

import os
import os.path as op
import unittest

import numpy as np

from divdet.surface_feature import SurfaceFeature

class Test(unittest.TestCase):
    """Test surface feature functionality"""

    def test_surface_feature(self):
        """Test that valid corners are calculated correctly when cropping"""

        # Fully valid image
        sf1 = SurfaceFeature(1, 1, 2, 2, 'dummy_wkt_string', 0.5, 'dummy_id')
        sf1.determine_quadkey()

        self.assertEqual(sf1.quadkey, '3000000')


if __name__ == '__main__':
    unittest.main()
