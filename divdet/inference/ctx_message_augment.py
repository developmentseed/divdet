import csv
from copy import copy

from absl import logging, app, flags
from tqdm import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('input_data_fpath', None, 'Filepath of input CSV containing CTX message info.')
flags.DEFINE_string('output_data_fpath', None, 'Filepath of output CSV that will be augmented with a url key.')



def main(_):
    all_row_data = []

    with open(FLAGS.input_data_fpath, 'r') as image_data_csv:
        csv_reader = csv.DictReader(image_data_csv)
        for row in tqdm(csv_reader, desc='Reading original CTX data'):
            temp_dict = row.copy()
            temp_dict['url'] = (f'https://image.mars.asu.edu/stream/'
                                f'{temp_dict["pds_id"]}.tiff?'
                                f'image=/mars/images/ctx/'
                                f'{temp_dict["volume_id"]}/prj_full/'
                                f'{temp_dict["pds_id"]}.tiff')
            all_row_data.append(temp_dict)

    with open(FLAGS.output_data_fpath, 'w') as image_data_csv:
        fieldnames = temp_dict.keys()
        csv_writer = csv.DictWriter(image_data_csv, fieldnames=fieldnames)

        csv_writer.writeheader()
        for row in tqdm(all_row_data, desc='Augmenting data with URLs'):
            csv_writer.writerow(row)


if __name__ == '__main__':
    app.run(main)
