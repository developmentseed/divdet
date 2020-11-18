"""
Construct a set of messages for Google PubSub from a CSV
"""
import csv
from copy import copy

from absl import logging, app, flags
from tqdm import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('input_data_fpath', None, 'Filepath of input CSV containing message info.')
flags.DEFINE_string('output_message_fpath', None, 'Filepath of output CSV that will have messages for pushing to PubSub.')

flags.DEFINE_string('csv_url_key', 'pds_jp2_url', 'Key to use to extract image URL from a CSV')
flags.DEFINE_string('csv_product_id_key', 'product_id', 'Key to use to extract image ID from a CSV')
flags.DEFINE_string('csv_version_id_key', None, 'Key to use to extract image version from a CSV')
flags.DEFINE_string('csv_volume_id_key', None, 'Key to use to extract image volume ID from a CSV')
flags.DEFINE_string('csv_lon_key', 'center_latitude', 'Key to use to extract image latitude from a CSV')
flags.DEFINE_string('csv_lat_key', 'center_longitude', 'Key to use to extract image longitude from a CSV')
flags.DEFINE_string('csv_projection_url_key', None, 'URL to download projection from.')

flags.DEFINE_string('csv_instrument_key', 'instrument_id', 'Key to use to extract the instrument ID from CSV')
flags.DEFINE_string('csv_instrument_host_key', None, 'Key to use to extract image the instrument host from CSV')
flags.DEFINE_string('csv_sub_solar_azimuth_key', 'sub_solar_azimuth', 'Key to use to extract subsolar aziumth from CSV')

flags.DEFINE_string('scales', '[0.25, 0.5, 1]', 'List of scales to use when multiscaling image for prediction.')
flags.DEFINE_integer('batch_size', 1, 'Number of windows to send to prediction endpoint at a time.')
flags.DEFINE_integer('window_size', 1024, 'Size of one side of a square to use for image windowing.')
flags.DEFINE_integer('min_window_overlap', 256, 'Number of pixels to overlap between windows.')
flags.DEFINE_string('prediction_endpoint', 'http://localhost:8501/v1/models/divdet:predict', 'Path to send images to for prediction.')
flags.DEFINE_bool('center_reproject', False, 'Whether or not to center reproject the image.')


def main(_):

    fdict = FLAGS.flag_values_dict()
    message_keys = [flag for flag in fdict.keys() if fdict.get(flag) and 'csv_' in flag]
    batch_info = dict(scales=FLAGS.scales,
                      batch_size=FLAGS.batch_size,
                      window_size=FLAGS.window_size,
                      min_window_overlap=FLAGS.min_window_overlap,
                      prediction_endpoint=FLAGS.prediction_endpoint,
                      center_reproject=FLAGS.center_reproject)


    row_data = []
    with open(FLAGS.input_data_fpath, 'r') as image_data_csv:
        csv_reader = csv.DictReader(image_data_csv)
        for row in tqdm(csv_reader, desc='Creating PubSub messages'):
            temp_dict = {}
            for key in message_keys:
                temp_dict[fdict[key]] = row[fdict[key]]
            row_data.append(temp_dict)

            '''
            row_data.append(dict(image_url=row[FLAGS.csv_url_key],
                                 product_id=row[FLAGS.csv_product_id_key],
                                 version_id=row.get(FLAGS.csv_version_id_key),
                                 center_lat=row[FLAGS.csv_lat_key],
                                 center_lon=row[FLAGS.csv_lon_key],
                                 instrument=row[FLAGS.csv_instrument_key],
                                 instrument_host=row[FLAGS.csv_instrument_host_key],
                                 projection_url=row[FLAGS.csv_projection_url_key],
                                 sub_solar_azimuth=row[FLAGS.csv_sub_solar_azimuth_key]))
            '''


    with open(FLAGS.output_message_fpath, 'w') as message_csv:
        # Unpack all key names
        fieldnames = [fdict[key] for key in message_keys] + [k for k in batch_info.keys()]

        csv_writer = csv.DictWriter(message_csv, fieldnames=fieldnames)
        csv_writer.writeheader()

        for temp_dict in row_data:
            temp_dict.update(batch_info)
            csv_writer.writerow(temp_dict)

if __name__ == "__main__":
    app.run(main)
