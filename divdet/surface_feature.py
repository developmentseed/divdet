"""
Class and tools for managing surface feature detections
"""

import os.path as op
import csv
import json
from pathlib import Path

import math
from osgeo import ogr
import numpy as np
from tqdm import tqdm

from sqlalchemy import create_engine, event, func
from sqlalchemy import (Column, Integer, String, Float,
                        ForeignKey)
from sqlalchemy.sql import select
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Index

from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_Intersects

# Set the declarative base to prep creation of SQL classes
Base = declarative_base()

# Constants
IOU_THRESH = 0.75
TILE_SIZE = 256
MAX_ZOOM = 20
MARS_RADIUS = 3396190  # From CTX data
EARTH_RADIUS = 6378137
ORIGIN_SHIFT = 2.0 * math.pi * MARS_RADIUS / 2.0
INITIAL_RESOLUTION = 2.0 * math.pi * MARS_RADIUS / float(TILE_SIZE)


def resolution(zoom):
    """Get the meters per pixel"""
    return INITIAL_RESOLUTION / (2 ** zoom)


def check_lonlat_validity(lat, lon):
    """Helper for error checking"""
    if abs(lat) > 90:
        raise RuntimeError(f'Latitude {lat} is outside valid range.')
    if abs(lon) > 180:
        raise RuntimeError(f'Longitude {lon} is outside valid range.')


class Crater(Base):
    """Geometry and properties for a single crater

    Attributes
    ----------
    geometry: str
        Polygon outlining the crater in WKT format.
    confidence: float
        ML model confidence that this is indeed a crater.
    eccentricity: float
        Eccentricity of the purported crater. Zero is a perfect circle.
        Ellipses are greater than zero but less than 1 with larger numbers
        corresponding to longer ellipses (larger difference between axes.
    gradient_angle: float
        Angle on the ground corresponding to the average angle from dark to
        light pixel intensity.
    """

    __tablename__ = 'craters'
    id = Column(Integer, primary_key=True)
    geometry = Column(Geometry('POLYGON'))
    confidence = Column(Float)
    eccentricity = Column(Float)
    gradient_angle = Column(Float)
    image_id = Column(Integer, ForeignKey('images.id'))

    image = relationship('Image', back_populates='craters')

    # Add table index according to
    #   https://stackoverflow.com/a/6627154
    __table_args__ = (Index('geom_index', 'geometry', 'confidence'), )

    def __repr__(self):
        """Define string representation."""
        return f'<Crater(confidence={self.confidence}, image={self.image})>'


class Image(Base):
    """Metadata for one satellite image

    Attributes
    ----------
    lon: float
        Longitude of the image center
    lat: float
        Latitude of the image center
    satellite: str
        Satellite platform (e.g., 'MRO' or 'LROC')
    camera: str
        Imaging platform (e.g., 'CTX', 'HiRISE', or 'WAC')
    pds_id: str
        Planetary Data System unique ID of the image (e.g., 'B04_011293_1265_XN_53S071W')
    subsolar_azimuth: float
        Direction of sun in image CCW from right.
    """

    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    lon, lat = Column(Float), Column(Float)
    pds_id = Column(String)
    satellite = Column(String)
    camera = Column(String)
    subsolar_azimuth = Column(Float)

    # Add a relationship with the Crater class
    craters = relationship('Crater', back_populates='image', cascade="all, delete, delete-orphan")

    def __repr__(self):
        """Define string representation."""
        return f'<Image({self.satellite}:{self.camera}, pds_id={self.pds_id})>'


@event.listens_for(session, 'before_flush')
def receive_before_flush(session, flush_context, instances):
    "Verify proposed database insertions using the 'before_flush' event."
    for proposed_object in session.new:

        ##############################
        # Look for repeat Image object
        if type(proposed_object) == Image:
            match = session.query(Image).filter(proposed_object.pds_id == Image.pds_id).one_or_none()
            if match:
                session.expunge(proposed_object)
                print(f'Aborted insert for {proposed_object}. Image ID exists already.')
            continue

        #########################################################################################
        # Identify if proposed craters should 1. not be inserted or 2. overwrite existing craters

        #stmt_confidence = proposed_object.confidence <= Crater.confidence
        stmt_any_overlap = func.ST_Intersects(proposed_object.geometry, Crater.geometry)  # include or no?

        # Conditions for match to existing crater (If met, skip insert)
        intersection = func.ST_Area(func.ST_Intersection(proposed_object.geometry, Crater.geometry))
        total_area = func.ST_Area(func.ST_Collect(proposed_object.geometry, Crater.geometry))
        iou = intersection / (total_area - intersection)

        # Run query against database
        matches = session.query(Crater).filter(stmt_any_overlap, iou > iou_thresh).all()
        if matches is None:
            continue

        # If crater overlaps found, determine if we should abort insert or overwrite.
        for crater_match in matches:
            print(f'Found {len(matches)} craters meeting IOU threshold...')

            # Abort insert because new detection is not higher than existing
            if proposed_object.confidence <= crater_match.confidence:
                session.expunge(proposed_object)
                print(f'Aborted insert for {proposed_object}.')
                return

            # Delete existing crater because new detection is higher confidence than existing
            else:
                session.delete(crater_match)
                print(f'Overwriting {crater_match}.')
