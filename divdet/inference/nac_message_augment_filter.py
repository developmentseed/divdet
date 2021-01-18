import csv
from copy import copy

import os.path as op
import numpy as np
from absl import logging, app, flags
from tqdm import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('input_data_fpath', None, 'Filepath of input NAC containing NAC message info.')
flags.DEFINE_string('output_data_fpath', None, 'Filepath of output CSV that will be augmented with a url key.')
flags.DEFINE_string('mission_phase', None, 'Phase to match in image metadata')
flags.DEFINE_float('latitude_threshold', None, 'Max absolute latitude')
flags.DEFINE_string('image_category', None, 'Image category to match (e.g., `NACL`')


def main(_):
    all_row_data = []

    with open(FLAGS.input_data_fpath, 'r') as image_data_csv:
        csv_reader = csv.DictReader(image_data_csv)
        for row in tqdm(csv_reader, desc='Reading original NAC data'):
            temp_dict = row.copy()

            if FLAGS.mission_phase:
                if temp_dict['mission_phase_name'] != FLAGS.mission_phase:
                    continue
            if FLAGS.image_category:
                if temp_dict['image_category'] != FLAGS.image_category:
                    continue
            if FLAGS.latitude_threshold:
                if np.abs(float(temp_dict['sub_spacecraft_latitude'])) > FLAGS.latitude_threshold:
                    continue

            file_spec_parts = temp_dict['file_specification_name'].split('/')

            temp_dict['url'] = op.join('https://pds.lroc.asu.edu/data/',
                                       file_spec_parts[0],
                                       file_spec_parts[1],
                                       'EXTRAS/BROWSE',
                                       file_spec_parts[4],
                                       f'{temp_dict["pds_id"]}_pyr.tif')
            all_row_data.append(temp_dict)

    with open(FLAGS.output_data_fpath, 'w') as image_data_csv:
        fieldnames = list(temp_dict.keys()) + ['url']
        csv_writer = csv.DictWriter(image_data_csv, fieldnames=fieldnames)

        csv_writer.writeheader()
        for row in tqdm(all_row_data, desc='Augmenting data with URLs'):
            csv_writer.writerow(row)


if __name__ == '__main__':
    app.run(main)
