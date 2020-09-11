"""
Functionality to perform inference. Acts as middle-man between image queue and
KF Serving cluster.

inference_runner.py
"""
import os.path as op
import time
import base64
import json
import tempfile
from io import BytesIO
import requests
from functools import partial
import ast
import logging

#import tensorflow as tf
import numpy as np
from tqdm import tqdm
from google.cloud import pubsub_v1
from google.oauth2 import service_account
#TODO: switch to rasterio if possible
from skimage.transform import rescale
#from skimage.io import imsave
from PIL import Image as PIL_Image
import rasterio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from absl import app, logging, flags
from osgeo import gdal

from divdet.inference.utils_inference import (
    iter_grouper, get_slice_bounds, yield_windowed_reads_numpy,
    windowed_reads_numpy, calculate_region_grad, calculate_shape_props,
    poly_non_max_suppression)
from divdet.surface_feature import Crater, Image, Base

flags.DEFINE_string('gcp_project', None, 'Google cloud project ID.')
flags.DEFINE_string('pubsub_subscription_name', None, 'Google cloud pubsub subscription name for queue.')
flags.DEFINE_string('inference_endpoint', None, 'Address of prediction cluster.')
#flags.DEFINE_integer('batch_size', 1, 'Number of images to run per inference batch.')
flags.DEFINE_string('database_uri', None, 'Address of database to store prediction results.')
flags.DEFINE_integer('max_outstanding_messages', 1, 'Number of messages to have in local backlog.')
flags.DEFINE_float('nms_iou', 0.5, 'Non-max suppression IOU threshold.')
flags.DEFINE_string('service_account_fpath', None, 'Filepath to service account with pubsub access.')


FLAGS = flags.FLAGS

overload_errors = ['<urlopen error [Errno 60] ETIMEDOUT>',
                   '<urlopen error [Errno 60] Operation timed out>',
                   '<urlopen error [Errno 2] Lookup timed out>',
                   '<urlopen error [Errno 8] Name or service not known>']


def download_url_to_file(url, directory='/tmp', clobber=False, repeat_tries=5,
                         chunk_size=10485760):
    """Download a url to disk

    Parameters
    ----------
    url: string
        URL to file to be downloaded
    directory: string
        Local directory to save file to.
    clobber: bool
        If file exists, should program overwrite it. Default is False
    repeat_tries: int
        Number of times to retry the URL download. Default is 5
    chunk_size: int
        Size of the download chunk size in bytes. Default is 10Mb

    Returns
    -------
    save_fpath: string
        Filepath to downloaded file on disk.
    """

    save_fpath = op.join(directory, url.split('/')[-1])
    if op.exists(save_fpath) and not clobber:
        raise ValueError(f'File exists at {save_fpath}')

    # Use `stream=True` for large files
    with requests.get(url, stream=True, allow_redirects=True) as req:
        try:
            req.raise_for_status()
            with open(save_fpath, 'wb') as write_file:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    if chunk: # filter out keep-alive new chunks
                        write_file.write(chunk)
            logging.info(f'Downloaded file from URL: {url}')
            return save_fpath

        # Handle some known exceptions
        except requests.exceptions.HTTPError as http_e:
            logging.error(f'HTTPError: {http_e}')

        except requests.exceptions.InvalidURL as url_e:
            logging.error(f'\nReceived invalid url error {url_e}')
            if str(url_e) in overload_errors:
                logging.error('Known load error, retrying')

        except Exception as err:
            logging.error(f'Other error on {url}')

        # If download was unsuccessful, retry a few times
        repeat_tries -= 1
        if repeat_tries == 0:
            logging.info(f'Too many repeats, stopping on {url}')


def arr_to_b64(numpy_arr):
    """Convert a numpy array into a b64 string"""

    # Generate image in bytes format; will convert using `tobytes` if not contiguous
    image_pil = PIL_Image.fromarray(numpy_arr)
    byte_io = BytesIO()
    image_pil.save(byte_io, format="PNG")

    #  Convert to base64; uses web-safe encoding
    b64_image = base64.urlsafe_b64encode(byte_io.getvalue()).decode("utf-8")
    byte_io.close()

    return b64_image


def pred_generator(generator, endpoint, batch_size=1):
    """Send a data from a generator to an inference endpoint"""

    ###################################
    # Create a batch of data to predict
    for image_batch in tqdm(iter_grouper(generator, batch_size)):

        pred_batch = []  # List to hold image metadata, prediction information
        instances = []  # List that will hold b64 images to be sent to endpoint

        for image_dict in image_batch:
            b64_image = arr_to_b64(image_dict['image_data'])
            #b64_image = arr_to_b64(image_dict.pop('image_data'))
            pred_batch.append(image_dict)

            #XXX KF Serving example here:
            #    https://github.com/kubeflow/kfserving/blob/master/docs/samples/tensorflow/input.json
            instances.append({'b64': b64_image})

        ################
        # Run prediction

        payload = json.dumps({"instances": instances})
        import ipdb; ipdb.set_trace()
        resp = requests.post(endpoint, data=payload)
        preds = json.loads(resp.content)['predictions']

        #############################
        # Store and return prediction
        for pred_dict, pred in zip(pred_batch, preds):
            pred_dict['prediction_data'] = pred

        yield pred_batch


def proc_message_debug(message, session, endpoint=None):
    logging.info('Received message: {}'.format(message))

    message.ack()


def proc_message(message, session):
    """Callback to  process an image"""
    logging.info('Received message: {}'.format(message.data))
    msg_dict = dict(message.attributes)

    if not message.attributes['scales']:
        msg_dict['scales'] = [1]
    else:
        msg_dict['scales'] = json.loads(msg_dict['scales'])

    msg_dict['window_size'] = int(msg_dict['window_size'])
    msg_dict['min_window_overlap'] = int(msg_dict['min_window_overlap'])
    msg_dict['projection_center_latitude'] = float(msg_dict['projection_center_latitude'])
    msg_dict['projection_center_longitude'] = float(msg_dict['projection_center_longitude'])
    msg_dict['sub_solar_azimuth'] = float(msg_dict['sub_solar_azimuth'])
    msg_dict['batch_size'] = int(msg_dict['batch_size'])

    # Use temp directory so everything is deleted after image processing completes
    with tempfile.TemporaryDirectory() as tmp_dir:
        ##############################################
        # Download data and reproject if needed
        ##############################################
        # TODO: Move back to more generic url key name
        logging.info('Downloading image: {}'.format(msg_dict['pds_jp2_url']))
        image_fpath_orig = download_url_to_file(msg_dict['pds_jp2_url'],
                                                directory=tmp_dir)

        '''
        if msg_dict['projection_url']:
            logging.info('Downloading projection.')
            proj_fpath = download_url_to_file(msg_dict['projection_url'],
                                              directory=tmp_dir)
        '''
        logging.info('Download complete.')

        if msg_dict['center_reproject'] == 'True':
            image_fpath = op.join(op.splitext(image_fpath_orig)[0] + '_warp',
                                  op.splitext(image_fpath_orig)[1])
            logging.info(f'Reprojecting image and saving to {image_fpath}.')

            with rasterio.open(image_fpath_orig) as temp_img:
                #center_lat = temp_img.lnglat()[1]  # Raster center
                center_lat = msg_dict['projection_center_lat']
            eqc_proj = f'+proj=eqc +lat_ts={center_lat} +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 +a=3396190 +b=3396190 +units=m +no_defs'
            gdal.Warp(image_fpath, image_fpath_orig, dstSRS=eqc_proj)
            logging.info('Reprojection complete.')

        else:
            image_fpath = image_fpath_orig

        ###########################################################
        # Slice image appropriately and pass to prediction endpoint
        ###########################################################
        preds = []
        slices = []
        with rasterio.open(image_fpath) as dataset:
            image = dataset.read(1)  # Read data from first band
            if image.dtype == np.uint16:
                image = (image / 65535. * 255).astype(np.uint8)

            for scale in msg_dict['scales']:
                logging.info('Processing at scale %s', scale)

                # Rescale image, round, and convert back to orig datatype
                scaled_image = rescale(image, scale, anti_aliasing=True).round().astype(image.dtype)

                # Calculate slice bounds and create generator
                slice_bounds = get_slice_bounds(
                    scaled_image.shape,
                    slice_size=(msg_dict['window_size'], msg_dict['window_size']) ,
                    min_window_overlap=(msg_dict['min_window_overlap'], msg_dict['min_window_overlap']))

                logging.info('Created %s slices.', (len(slice_bounds)))
                #slice_batch = yield_windowed_reads_numpy(scaled_image, slice_bounds)
                slice_batch = windowed_reads_numpy(scaled_image, slice_bounds)

                # Calculate slice bounds and create generator
                pred_gen = pred_generator(slice_batch,
                                          msg_dict['prediction_endpoint'],
                                          msg_dict['batch_size'])

                preds.extend(list(pred_gen))
                logging.info(f'Finished processing at scale {scale}')
                slices.extend(list(slice_batch))

    ###########################
    # Run non-max suppression to remove duplicates within one image

    # Non-max suppression for bounding boxes only
    #selected_inds = tf.image.non_max_suppression(boxes, scores,
    #                                             max_output_size=len(preds))

    # Run non-max suppression that uses crater polygon masks
    preds_polygons = [pred['mask'] for pred in preds]
    preds_confs = [pred['confidence'] for pred in preds]

    selected_inds = poly_non_max_suppression(preds_polygons, preds_confs)

    preds = [preds[ind] for ind in selected_inds]
    slices = [slices[ind] for ind in selected_inds]

    ###########################
    # Loop over preds, determine properties, and store
    for (pred, mask), img_slice in zip(preds, slices):

        # TODO: will need to bring along original image for gradient calcs
        #grad_h, grad_v = calculate_region_grad(mask)
        shape_props = calculate_shape_props(mask)

        # Pass results back to database
        session.add(Image(lat=msg_dict['projection_center_latitude'],
                          lon=msg_dict['projection_center_longitude'],
                          instrument_host_id=msg_dict['instrument_host_id'],
                          instrument_id=msg_dict['instrument_id'],
                          pds_id=msg_dict['product_id'],
                          pds_version_id=msg_dict['pds_version_id'],
                          subsolar_azimuth=msg_dict['sub_solar_azimuth']))

        # TODO: update grad angle and image ID
        session.add(Crater(geometry=pred['geom'],
                           confidence=pred['confidence'],
                           eccentricity=shape_props['eccentricity'],
                           gradient_angle=-1,
                           image_ID=-1))

    session.commit()
    message.ack()


def _get_session(db_uri, use_batch_mode=True, echo=False):
    """Helper to get an SQLAlchemy DB session"""
    # `use_batch_mode` is experimental currently, but needed for `executemany`
    #engine = create_engine(db_uri, use_batch_mode=use_batch_mode, echo=echo)
    engine = create_engine(db_uri, echo=echo)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        connection = session.connection()
        logging.info('Successfully connected to database.')
    except:
        raise RuntimeError(f'Couldn\'t connect to db: {db_uri}')

    return session


def main(_):

    # Setup pubsub subscriber
    if FLAGS.service_account_fpath:
        credentials = service_account.Credentials.from_service_account_file(
            FLAGS.service_account_fpath,
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
        subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
    else:
        subscriber = pubsub_v1.SubscriberClient()

    # Set relevant subscription params
    subscription_path = subscriber.subscription_path(FLAGS.gcp_project,
                                                     FLAGS.pubsub_subscription_name)
    flow_control = pubsub_v1.types.FlowControl(
        max_messages=FLAGS.max_outstanding_messages)
    # Can block the asynchronous call by using the `future` variable returned

    session = _get_session(FLAGS.database_uri)

    # XXX To extend message processing beyond 10 mins, see https://github.com/googleapis/google-cloud-go/issues/608
    debug_call_func = partial(proc_message_debug, session=session)
    call_func = partial(proc_message, session=session)
    subscriber.subscribe(subscription_path,
                         callback=call_func,
                         flow_control=flow_control)#, ack_deadline_seconds=600)
    logging.info(f'Listening for messages on {subscription_path}')

    # XXX: Eventually, need to remove this for pod exit
    while True:
        time.sleep(10)

    logging.info('Finished inference.')


if __name__ == '__main__':
    app.run(main)
