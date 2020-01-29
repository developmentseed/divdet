"""
Test utilities component of divdet
"""

import os
import os.path as op
import unittest

import numpy as np

from divdet.utils_data import (_get_valid_corners)

class Test(unittest.TestCase):
    """Test utilities functionality"""

    def test_valid_corners(self):
        """Test that valid corners are calculated correctly when cropping"""

        # Fully valid image
        input_img = np.ones((4, 4))
        ret = _get_valid_corners(input_img, crop_size=3, l_thresh=1,
                                 corner_thresh=0.5)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret, np.ones((2, 2)))

        # Image doesn't meet saturation level threshold
        ret = _get_valid_corners(input_img, crop_size=3, l_thresh=2,
                                 corner_thresh=0.5)
        self.assertEqual(ret, np.array([[False, False], [False, False]]))

        # Right half is zeros but still above threshold
        input_img[:, 2:] = 0
        ret = _get_valid_corners(input_img, crop_size=3, l_thresh=1,
                                 corner_thresh=0.3)
        self.assertEqual(ret, np.array([[True, True], [True, True]]))

        # Right half is zeros and below threshold
        ret = _get_valid_corners(input_img, crop_size=3, l_thresh=1,
                                 corner_thresh=0.4)
        self.assertEqual(ret, np.array([[True, False], [True, False]]))


if __name__ == '__main__':
    unittest.main()