
# This script creates TFRecord files for the testing data set.

import tensorflow as tf
import cv2
import glob2
import math

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Path to the directory containing test TIFF images.
image_paths = glob2.glob('../test-tif-v2/*')
temp = image_paths[0].split('/')
print(temp)

# Path for the directory to store TFRecord files for test data.
tfrecord_path_prefix = '../tfrecords/test/'

num_shards = 6
records_per_file = math.ceil(len(image_paths)/num_shards)

tfrecord_paths = [tfrecord_path_prefix+'planet-'+str(i*records_per_file)+'-'+str((i+1)*records_per_file)+'.tfrecord'
                  if i!=num_shards-1
                  else tfrecord_path_prefix+'planet-'+str(i*records_per_file)+'-'+str(len(image_paths))+'.tfrecord'
                  for i in range(num_shards)]

print(tfrecord_paths)

tfrecord_writer = None
num_processed = 0
for i in range(num_shards):
    tfrecord_writer = tf.io.TFRecordWriter(tfrecord_paths[i])
    image_paths_shard = image_paths[i*records_per_file: (i+1)*records_per_file] if i!=num_shards-1 \
                        else image_paths[i*records_per_file: len(image_paths)]
    for image_path in image_paths_shard:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #encode
        image_string = image.tostring()
        feature_dict = {'image_string':_bytes_feature(image_string)}
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        tfrecord_writer.write(example.SerializeToString())
    num_processed = num_processed + len(image_paths_shard)
    print('Processed: '+str(num_processed)+ ' files.')

tfrecord_writer.close()
