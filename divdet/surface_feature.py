"""
Class and tools for managing surface feature detections
"""

import math

MARS_RADIUS = 3396190  # From CTX data
EARTH_RADIUS = 6378137
TILE_SIZE = 256
ORIGIN_SHIFT = 2.0 * math.pi * MARS_RADIUS / 2.0
INITIAL_RESOLUTION = 2.0 * math.pi * MARS_RADIUS / float(TILE_SIZE)
MAX_ZOOM = 20


class SurfaceFeature(object):
    """Object to hold information about a detected surface feature"""

    def __init__(self, tl_lon, tl_lat, br_lon, br_lat, poly, confidence,
                 image_id):
        """Surface feature containing info, geometry, etc. for a surface detection

        Parameters
        ----------
        tl_lon: float
            Top left longitude
        tl_lat: float
            Top left latitude
        br_lon: float
            Bottom right longitude
        br_lat: float
            Bottom right latitude
        poly: str
            Polygon outlining feature in WKT format
        confidence: float
            Confidence of prediction on interval [0, 1]
        image_id: str
            Unique ID of the image (e.g., "B04_011293_1265_XN_53S071W")
        """

        check_lonlat_validity(tl_lon, tl_lat)
        check_lonlat_validity(br_lon, br_lat)
        self.tl_lon, self.tl_lat = tl_lon, tl_lat
        self.br_lon, self.br_lat = br_lon, br_lat
        self.poly = poly

        if not 0 <= confidence <= 1:
            raise ValueError(f'Confidence should be on interval [0, 1]. Got {confidence}')
        self.confidence = confidence

        self.image_id = image_id
        self.quadkey = None

    def determine_quadkey(self):
        """Set the quadkey for a surface feature based on TL and BR points"""
        qk1 = point_to_quadkey(self.tl_lon, self.tl_lat, MAX_ZOOM)
        qk2 = point_to_quadkey(self.br_lon, self.br_lat, MAX_ZOOM)

        self.quadkey = smallest_common_quadkey(qk1, qk2)


def resolution(zoom):
    """Get the meters per pixel"""
    return INITIAL_RESOLUTION / (2 ** zoom)


def point_to_quadkey(longitude, latitude, zoom):
    """Convert a point to a quadkey string"""

    meter_x = longitude * ORIGIN_SHIFT / 180.0
    meter_y = math.log(math.tan((90.0 + latitude) * math.pi / 360.0)) / (math.pi / 180.0)
    meter_y = meter_y * ORIGIN_SHIFT / 180.0

    pixel_x = abs(round((meter_x + ORIGIN_SHIFT) / resolution(zoom=zoom)))
    pixel_y = abs(round((meter_y - ORIGIN_SHIFT) / resolution(zoom=zoom)))

    tms_x = int(math.ceil(pixel_x / float(TILE_SIZE)) - 1)
    tms_y = int(math.ceil(pixel_y / float(TILE_SIZE)) - 1)
    tms_y = (2 ** zoom - 1) - tms_y

    value = ''
    #tms_y = (2 ** self.zoom - 1) - tms_y
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tms_x & mask) != 0:
            digit += 1
        if (tms_y & mask) != 0:
            digit += 2
        value += str(digit)
    return value


def smallest_common_quadkey(qk1, qk2):
    """Find the smallest quadkey that overlaps two points."""

    for ind, (k1, k2) in enumerate(zip(qk1, qk2)):
        if k1 == k2:
            continue
        else:
            return qk1[:ind]

    return ''


def check_lonlat_validity(lat, lon):
    """Helper for error checking"""
    if abs(lat) > 90:
        raise RuntimeError(f'Latitude {lat} is outside valid range.')
    if abs(lon) > 180:
        raise RuntimeError(f'Longitude {lon} is outside valid range.')
