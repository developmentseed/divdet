"""
Functionality to perform inference. Acts as runner between image queue and
GPU cluster containing trained models.

inference_runner.py
"""
import os
import os.path as op
import time
import base64
import json
import tempfile
from io import BytesIO
from functools import partial
import logging

import requests
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from skimage.transform import rescale, resize
from PIL import Image as PIL_Image
import rasterio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from absl import app, logging, flags
from osgeo import gdal

from divdet.inference.utils_inference import (
    iter_grouper, get_slice_bounds, windowed_reads_numpy,
    calculate_region_grad, calculate_shape_props,
    poly_non_max_suppression, convert_mask_to_polygon,
    geospatial_polygon_transform)
from divdet.surface_feature import Crater, Image, Base

flags.DEFINE_string('gcp_project', None, 'Google cloud project ID.')
flags.DEFINE_string('pubsub_subscription_name', None, 'Google cloud pubsub subscription name for queue.')
flags.DEFINE_string('database_uri', None, 'Address of database to store prediction results.')
flags.DEFINE_integer('max_outstanding_messages', 1, 'Number of messages to have in local backlog.')
flags.DEFINE_string('service_account_fpath', None, 'Filepath to service account with pubsub access.')


FLAGS = flags.FLAGS

# Known errors that occur during image download step
overload_errors = ['<urlopen error [Errno 60] ETIMEDOUT>',
                   '<urlopen error [Errno 60] Operation timed out>',
                   '<urlopen error [Errno 2] Lookup timed out>',
                   '<urlopen error [Errno 8] Name or service not known>']


def download_url_to_file(url, directory='/tmp', clobber=False, repeat_tries=5,
                         chunk_size=2097152):
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
    with requests.get(url, stream=True, allow_redirects=True, timeout=10) as req:
        try:
            req.raise_for_status()
            with open(save_fpath, 'wb') as write_file:
                pbar = tqdm(total=int(req.headers['Content-Length']))

                for chunk in req.iter_content(chunk_size=chunk_size):
                    if chunk: # filter out keep-alive new chunks
                        write_file.write(chunk)
                        pbar.update(len(chunk))
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
            logging.error(f'Other error on {url}: {err}')

        # If download was unsuccessful, retry a few times
        repeat_tries -= 1
        if repeat_tries == 0:
            logging.info(f'Too many repeats, stopping on {url}')

        if op.exists(save_fpath):
            os.remove(save_fpath)


def arr_to_b64(numpy_arr, ensure_RGB=True):
    """Convert a numpy array into a b64 string"""

    # Generate image in bytes format; will convert using `tobytes` if not contiguous
    image_pil = PIL_Image.fromarray(numpy_arr)
    if ensure_RGB:
        image_pil = image_pil.convert('RGB')

    byte_io = BytesIO()
    image_pil.save(byte_io, format='PNG')

    # Convert to base64
    b64_image = base64.b64encode(byte_io.getvalue()).decode('utf-8')

    # Web-safe encoding (haven't had much luck with this yet -- fails to decode)
    #b64_image = b64_image.replace('+', '-')
    #b64_image = b64_image.replace('/', '_')
    #b64_image = base64.urlsafe_b64encode(byte_io.getvalue()).decode("utf-8")
    byte_io.close()

    return b64_image


def pred_generator(generator, endpoint):
    """Send a data from a generator to an inference endpoint"""

    ###################################
    # Create a batch of data to predict
    for image_dict in tqdm(generator, desc='Making inference requests.'):

        b64_image = arr_to_b64(image_dict['image_data'])

        #XXX KF Serving example here:
        #    https://github.com/kubeflow/kfserving/blob/master/docs/samples/tensorflow/input.json
        instances = [{'b64': b64_image}]

        ################
        # Run prediction

        payload = json.dumps({"inputs": {"input_tensor": instances}})  # TF "col" format

        resp = requests.post(endpoint, data=payload)
        resp_json = json.loads(resp.content)
        if 'outputs' in resp_json.keys():
            resp_outputs = resp_json['outputs']
        else:
            logging.error(f'Error in prediction step. Raw json: {resp_json}')

        #############################
        # Store and return prediction

        # Only keep prediction indicies with confidences > 0 (some may have been filtered by OD API config)
        n_good_inds = int(np.sum(np.array(resp_outputs['detection_scores'][0]) > 0))

        image_dict['detection_scores'] = resp_outputs['detection_scores'][0][:n_good_inds]  # Probability each proposal corresponds to an object
        image_dict['detection_masks'] = resp_outputs['detection_masks'][0][:n_good_inds]  # Mask for each proposal. Needs to be resized from (33, 33)
        image_dict['proposal_boxes'] = resp_outputs['proposal_boxes'][0][:n_good_inds]  # Box coordinates for each object in orig image coords

        yield image_dict


def pred_generator_batched(generator, endpoint, batch_size=1):
    """Send a data from a generator to an inference endpoint"""

    ###################################
    # Create a batch of data to predict
    for image_batch in tqdm(iter_grouper(generator, batch_size)):

        pred_batch = []  # List to hold image metadata, prediction information
        instances = []  # List that will hold b64 images to be sent to endpoint

        for image_dict in image_batch:
            if image_dict is None:
                continue
            b64_image = arr_to_b64(image_dict['image_data'])
            #b64_image = arr_to_b64(image_dict.pop('image_data'))
            pred_batch.append(image_dict)

            #XXX KF Serving example here:
            #    https://github.com/kubeflow/kfserving/blob/master/docs/samples/tensorflow/input.json
            instances.append({"b64": b64_image})

        ################
        # Run prediction

        #payload = json.dumps({"instances": instances})  # TF "row" format
        payload = json.dumps({"inputs": {"input_tensor": instances}})  # TF "col" format

        # Sometimes, the prediction endpoint chokes. Allow for multiple retries
        retries = 5
        while retries:
            try:
                resp = requests.post(endpoint, data=payload, timeout=10)
                resp_outputs = json.loads(resp.content)['outputs']
                break
            except Exception as e:
                retries -= 1

                if retries == 0:
                    logging.error('Problem in prediction step.')
                    if resp:
                        del resp
                    raise e
                time.sleep(1)
        #############################
        # Store and return prediction
        for pi, pred_dict in enumerate(pred_batch):

            n_good_inds = int(np.sum(np.array(resp_outputs['detection_scores'][pi]) > 0))  # Throw out any 0-confidence predictions
            pred_dict['detection_scores'] = resp_outputs['detection_scores'][pi][:n_good_inds]  # Probability each proposal corresponds to an object
            pred_dict['detection_masks'] = resp_outputs['detection_masks'][pi][:n_good_inds]  # Mask for each proposal. Needs to be resized from (33, 33)
            pred_dict['proposal_boxes'] = resp_outputs['proposal_boxes'][pi][:n_good_inds]  # Box coordinates for each object in orig image coords

        yield pred_batch


def proc_message_debug(message, session, endpoint=None):
    logging.info('\nReceived message: {}'.format(message))

    message.ack()


def proc_message(message, session):
    """Callback to  process an image"""
    start_time = time.time()
    logging.info('\nReceived message: {}'.format(message.data))
    msg_dict = dict(message.attributes)

    if not message.attributes['scales']:
        msg_dict['scales'] = [1]
    else:
        msg_dict['scales'] = json.loads(msg_dict['scales'])


    # Check if image predictions already exists:
    skip_run_check = session.query(Image.id).\
                        filter(Image.pds_id == msg_dict['pds_id']).first()
    if skip_run_check:
        logging.info(f'Image {msg_dict["pds_id"]} already exists in DB. Acknowledging message and skipping.')
        message.ack()
        return

    msg_dict['window_size'] = int(msg_dict['window_size'])
    msg_dict['min_window_overlap'] = int(msg_dict['min_window_overlap'])
    msg_dict['center_latitude'] = float(msg_dict['center_latitude'])
    msg_dict['center_longitude'] = float(msg_dict['center_longitude'])
    msg_dict['sub_solar_azimuth'] = float(msg_dict['sub_solar_azimuth'])
    msg_dict['batch_size'] = int(msg_dict['batch_size'])

    # Use temp directory so everything is deleted after image processing completes
    with tempfile.TemporaryDirectory() as tmp_dir:
        ##############################################
        # Download data and reproject if needed
        ##############################################
        # TODO: Move back to more generic url key name
        logging.info('\nDownloading image: {}'.format(msg_dict['url']))
        image_fpath_orig = download_url_to_file(msg_dict['url'],
                                                directory=tmp_dir)
        if image_fpath_orig is None:
            logging.error('Image download errored.')
            message.nack()
            return

        try:
            message.modify_ack_deadline(600)
            logging.info('Updating acknowledgement deadline.')
        except:
            logging.info('Mod to ack deadline failed. Skipping')
        '''
        if msg_dict['projection_url']:
            logging.info('Downloading projection.')
            proj_fpath = download_url_to_file(msg_dict['projection_url'],
                                              directory=tmp_dir)
        '''

        if msg_dict['center_reproject'] == 'True':
            image_fpath = op.splitext(image_fpath_orig)[0] + '_warp' + \
                op.splitext(image_fpath_orig)[1]
            logging.info(f'Reprojecting image and saving to {image_fpath}.')

            with rasterio.open(image_fpath_orig) as temp_img:
                #center_lat = temp_img.lnglat()[1]  # Raster center
                center_lat = msg_dict['center_latitude']
            if msg_dict['instrument_host_id'] == 'CTX':
                # Mars projection
                eqc_proj = f'+proj=eqc +lat_ts={center_lat} +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 +a=3396190 +b=3396190 +units=m +no_defs'
                gdal.Warp(image_fpath, image_fpath_orig, dstSRS=eqc_proj)
                logging.info('Reprojection complete.')
            elif msg_dict['instrument_host_id'] == 'LRO':
                # Moon projection
                # Could also match the latitude to the images center here
                eqc_proj = '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs'
                gdal.Warp(image_fpath, image_fpath_orig, dstSRS=eqc_proj)
                logging.info('Reprojection complete.')

        else:
            image_fpath = image_fpath_orig

        ###########################################################
        # Slice image appropriately and pass to prediction endpoint
        ###########################################################
        preds = {'detection_scores': [], 'detection_masks': [], 'proposal_boxes': [],
                 'polygons': [], 'resized_masks': []}
        #slices = []
        with rasterio.open(image_fpath) as dataset:
            image = dataset.read(1)  # Read data from first band
            if image.dtype == np.uint16:
                image = (image / 65535. * 255).astype(np.uint8)
            if dataset.transform == rasterio.Affine.identity():
                affine_transform = rasterio.Affine(1, 0, 0, 0, -1, 0)  # If not mapped, use this transform for correct matching of image/crater coords
            else:
                affine_transform = dataset.transform


            for scale in msg_dict['scales']:
                logging.info('Processing at scale %s', scale)
                try:
                    message.modify_ack_deadline(600)
                    logging.info('Updated acknowledgement deadline.')
                except:
                    logging.info('Mod to ack deadline failed. Skipping')

                # Rescale image, round, and convert back to orig datatype

                scaled_image = rescale(image, scale, mode='edge', preserve_range=True,
                                       anti_aliasing=True).round().astype(image.dtype)

                # Calculate slice bounds and create generator
                slice_bounds = get_slice_bounds(
                    scaled_image.shape,
                    slice_size=(msg_dict['window_size'], msg_dict['window_size']),
                    min_window_overlap=(msg_dict['min_window_overlap'],
                                        msg_dict['min_window_overlap']))

                if not slice_bounds:
                    continue

                logging.info('Created %s slices. Running predictions at scale %s',
                            (len(slice_bounds)), scale)
                slice_batch = windowed_reads_numpy(scaled_image, slice_bounds)

                # Generate predictions. Use batched prediction if desired
                if msg_dict['batch_size'] > 1:
                    pred_gen = pred_generator_batched(slice_batch,
                                                      msg_dict['prediction_endpoint'],
                                                      msg_dict['batch_size'])

                    start_inf_time = time.time()
                    pred_batch = [item for sublist in pred_gen for item in sublist]
                    elapsed_time = time.time() - start_inf_time
                    avg_sec = elapsed_time / len(pred_batch)
                    logging.info(f'Avg image processing ({len(pred_batch)} images): {avg_sec:0.3f}s')

                else:
                    pred_gen = pred_generator(slice_batch,
                                              msg_dict['prediction_endpoint'])
                    pred_batch = list(pred_gen)
                logging.info('Crater predictions at scale %s complete.', (scale))

                # Convert predictions to polygon in orig image coordinate frame
                for pred_set, slice_set in tqdm(zip(pred_batch, slice_bounds),
                                                desc='\tConvert slice pred batch masks to polygons'):
                    y_offset_img = np.int(slice_set[0] / scale)
                    x_offset_img = np.int(slice_set[1] / scale)

                    pred_set['polygons'] = []
                    pred_set['resized_masks'] = []
                    new_proposal_boxes = []

                    for mask, box in zip(pred_set['detection_masks'], pred_set['proposal_boxes']):

                        # Get width/height of image and compute offset of slice
                        width = np.int(np.max((np.around((box[3] - box[1]) / scale), 1)))
                        height = np.int(np.max((np.around((box[2] - box[0]) / scale), 1)))

                        x_offset_box = box[1] / scale
                        y_offset_box = box[0] / scale
                        x_offset = x_offset_img + x_offset_box
                        y_offset = y_offset_img + y_offset_box

                        box = [y_offset, x_offset, y_offset+height, x_offset+width]  # Update proposal_boxes
                        new_proposal_boxes.append(box)

                        # Don't resize if scale is 1
                        if scale != 1:
                            mask_resized = resize(np.array(mask), (height, width),
                                                  mode='edge', anti_aliasing=True)
                        else:
                            mask_resized = np.array(mask)
                        mask_binary = mask_resized > 0.5  # Must be binary
                        pred_set['resized_masks'].append(mask_binary.astype(np.int))

                        # Generate polygon (with geospatial offset) from mask
                        mask_poly = convert_mask_to_polygon(mask_binary, (x_offset, y_offset))
                        pred_set['polygons'].append(mask_poly)  # polygon in whole-image pixel coordinates
                    pred_set['proposal_boxes'] = new_proposal_boxes

                    for key in ['detection_scores', 'detection_masks',
                                'proposal_boxes', 'polygons', 'resized_masks']:
                        preds[key].extend(pred_set[key])

                logging.info(f'Finished processing at scale {scale}')

            ###########################
            # Run non-max suppression to remove duplicates in multiple scales of one image

            logging.info(f"Found {len(preds['polygons'])} polygon predictions. Starting bbox-based NMS.")
            if len(preds['polygons']):

                # Non-max suppression for bounding boxes only
                selected_inds = tf.image.non_max_suppression(preds['proposal_boxes'],
                                                             preds['detection_scores'],
                                                             iou_threshold=0.2,
                                                             max_output_size=len(preds['detection_scores'])).numpy()

                # Select data from TF Serving column format
                for key in ['detection_scores', 'detection_masks', 'proposal_boxes', 'resized_masks']:
                    preds[key] = [preds[key][ind] for ind in selected_inds]

                # Convert polygons from pixel coords to geospatial coords
                preds['polygons'] = [geospatial_polygon_transform(preds['polygons'][ind], affine_transform)
                                     for ind in selected_inds]
                logging.info(f"After NMS, {len(preds['polygons'])} predictions remain.")

        try:
            message.modify_ack_deadline(600)
            logging.info('Updating acknowledgement deadline.')
        except:
            logging.info('Mod to ack deadline failed. Skipping')

        ###########################
        # Save image and craters to DB
        logging.info("Inserting %s polygon predictions.", len(preds['polygons']))

        # Save image first to get the image ID
        image_obj = Image(lon=msg_dict['center_longitude'],
                          lat=msg_dict['center_latitude'],
                          instrument_host_id=msg_dict.get('instrument_host_id', 'None'),
                          instrument_id=msg_dict['instrument_id'],
                          pds_id=msg_dict['pds_id'],
                          pds_version_id=msg_dict.get('pds_version_id', 'None'),
                          sub_solar_azimuth=msg_dict['sub_solar_azimuth'])
        session.add(image_obj)
        session.commit()

        # Loop over predicted craters, determine properties, and store
        for pi in range(len(preds['detection_scores'])):

            # Resize mask to original prediction bbox dimensions

            # TODO: will need to bring along original image for gradient calcs
            #grad_h, grad_v = calculate_region_grad(preds['resized_masks'][pi].astype(np.bool))
            shape_props = calculate_shape_props(preds['resized_masks'][pi])

            export_geom = preds['polygons'][pi].ExportToWkt()
            session.add(Crater(geometry=export_geom,
                               confidence=preds['detection_scores'][pi],
                               eccentricity=shape_props['eccentricity'],
                               gradient_angle=-1,
                               image_id=image_obj.id))

        session.commit()

    message.ack()

    elapsed_time = time.time() - start_time
    logging.info('***Processing complete *** %s', msg_dict["url"])
    logging.info('Total processing time: %s',
                 time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))


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
    # Get database connection
    time.sleep(10)  # Wait a bit to give DB proxy container time to start

    # Setup pubsub subscriber client
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
    # Add flow control to keep from downloading too many messages at once
    flow_control = pubsub_v1.types.FlowControl(
        max_messages=FLAGS.max_outstanding_messages)

    # Get DB session
    session = _get_session(FLAGS.database_uri)

    # Set callback for processing each message and begin pulling messages
    callback = partial(proc_message, session=session)
    streaming_pull_future = subscriber.subscribe(subscription_path,
                                                 callback=callback,
                                                 flow_control=flow_control)
    with subscriber:
        try:
            streaming_pull_future.result()
        except TimeoutError:
            streaming_pull_future.cancel()
            logging.info('Timeout error on {subscription_path}')

    logging.info('\nFinished inference.')


if __name__ == '__main__':
    app.run(main)
