import os
import math

import csv
import configparser as cp
import tensorflow as tf
from collections import Iterable


class Extractor:

    def __init__(self,
               files,
               parse_func=None,
               ):
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        cfg = cp.ConfigParser()
        cfg.read(os.path.dirname(self.root_dir) + '\config.ini')
        self.config = cfg['DEFAULT']
        self.config_dt = cfg['DATA_TABLE']
        # cast un-input variables
        if parse_func is None:
            parse_func = self._deserialize_example

        # check input variables
        assert isinstance(files, Iterable), 'files_names must be iterable.'
        for f in files:
            assert 'name' in f and 'type' in f, \
                "files_names's element must be a dict contains file's name and file's type."
        assert callable(parse_func), 'parse_func must be a function.'

        # import configuration
        self.COLUMNS = self.config_dt.get('COLUMNS').split(',')
        self.INT_FEATURE = self.config_dt.get('INT_FEATURE').split(',')
        self.FLOAT_FEATURE = self.config_dt.get('FLOAT_FEATURE').split(',')
        self.STRING_FEATURE = self.config_dt.get('STRING_FEATURE').split(',')
        self.FEATURE = self.INT_FEATURE + self.FLOAT_FEATURE + self.STRING_FEATURE
        self.TARGET = self.config_dt.get('TARGET').split(',')
        self.UNUSED_FEATURE = list(set(self.COLUMNS) - set(self.FEATURE) - set(self.TARGET))

        self.data_directory = self.config.get('DATA_DIRECTORY')

        # whether to convert existing file(s) into TFRecord format
        if self.config.getboolean('CONVERT_TO_TFRECORD'):
            self._rewrite_as_tfrecord(files)

        # creating dataset from file(s)
        files = ['{}\\{}\\{}.tfrecord'.format(self.root_dir, self.data_directory, f['name']) for f in files]
        for f in files:
            assert os.path.isfile(f), '{} does not exist'.format(f)
        # files = tf.data.Dataset.list_files(config.get('DATA_DIRECTORY') + files)
        dataset = tf.data.TFRecordDataset(
            files,
            num_parallel_reads=max(len(files), math.floor(os.cpu_count() / 2))
        )
        if len(files) > 1:
            dataset = dataset.interleave(
                map_func=parse_func,
                num_parallel_calls=math.floor(os.cpu_count() / 2),
                cycle_length=self.config.getint('CYCLE_LENGTH')
            )
        dataset = dataset.shuffle(buffer_size=self.config.getint('BUFFER_SIZE'))
        dataset = dataset.repeat(count=self.config.getint('NO_EPOCH'))
        dataset = dataset.batch(batch_size=self.config.getint('BATCH_SIZE'))
        if len(files) == 1:
            dataset = dataset.map(map_func=parse_func, num_parallel_calls=math.floor(os.cpu_count() / 2))
        self._dataset = dataset.prefetch(buffer_size=self.config.getint('PREFETCH_BUFFER_SIZE'))

    @property
    def dataset(self):
        return self._dataset

    def _deserialize_example(self, e):
        feature_description = {}
        for feature_name in self.COLUMNS:
            if feature_name in self.UNUSED_FEATURE:
                continue
            if feature_name in self.INT_FEATURE:
                feature_description[feature_name] = tf.FixedLenFeature([], tf.int64, default_value=0.0)
            elif feature_name in self.FLOAT_FEATURE + self.TARGET:
                feature_description[feature_name] = tf.FixedLenFeature([], tf.float32, default_value=0.0)
            elif feature_name in self.STRING_FEATURE:
                feature_description[feature_name] = tf.FixedLenFeature([], tf.string, default_value='')
        if self.config.getint('BATCH_SIZE') == 1:
            return tf.parse_single_example(e, feature_description)
        return tf.parse_example(e, feature_description)
        # return e

    # create a record
    def _serialize_example(self, row):
        example = tf.train.Example()
        # assign corresponding data type
        for i in range(len(self.COLUMNS)):
            feature_name = self.COLUMNS[i]
            feature_value = row[i]
            if feature_name in self.UNUSED_FEATURE:
                continue
            if feature_name in self.INT_FEATURE:
                example.features.feature[feature_name].int_list.value.extend([int(feature_value)])
            elif feature_name in self.FLOAT_FEATURE + self.TARGET:
                example.features.feature[feature_name].float_list.value.extend([float(feature_value)])
            elif feature_name in self.STRING_FEATURE:
                example.features.feature[feature_name].bytes_list.value.extend([bytes(feature_value, 'utf-8')])

        return example

    # create an iterative list of opened file
    def _create_csv_iterator(self, csv_file_path, skip_header):
        with tf.gfile.Open(csv_file_path) as csv_file:
            reader = csv.reader(csv_file)
            if skip_header:  # Skip the header
                next(reader)
            for row in reader:
                yield row

    # convert input file(s) into TFRecord format
    def _rewrite_as_tfrecord(self, files_name):
        output_tfrecord_files = ['{}\\{}\\{}.tfrecord'.format(self.root_dir, self.data_directory, f['name']) for f in
                                 files_name]
        for output_file, source_file in zip(output_tfrecord_files, files_name):
            writer = tf.python_io.TFRecordWriter(output_file)

            print("Creating TFRecords file at", output_file, "...")
            for i, row in enumerate(
                    self._create_csv_iterator(
                        '{}\\{}\\{}.{}'.format(self.root_dir, self.data_directory, source_file['name'],
                                               source_file['type']),
                        skip_header=True)):
                if len(row) == 0:
                    continue

                e = self._serialize_example(row)
                content = e.SerializeToString()
                writer.write(content)

            writer.close()

        print("Finish Writing", output_tfrecord_files)


if __name__ == '__main__':
    cfg = cp.ConfigParser()
    cfg.read('./config.ini')
    config = cfg['DEFAULT']
    extractor = Extractor(
        [{'name': 'SP500', 'type': 'csv'}],
    )
