
# This script creates TFRecord files for training data set.

import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import math

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Path for TIFF training images directory
images_path_prefix = '../train-tif-v2/'

# Path for directory to store TFRecord files
tfrecord_path_prefix = '../tfrecords/train/'

# Path for train_v2.csv file
data = pd.read_csv('train_v2.csv')
tags = data['tags'].str.split(' ').values
image_ids = data['image_name']
num_shards = 4 #set
records_per_file = math.ceil(len(image_ids)/num_shards)

tfrecord_paths = [tfrecord_path_prefix+'planet-'+str(i*records_per_file)+'-'+str((i+1)*records_per_file)+'.tfrecord'
                  if i!=num_shards-1
                  else tfrecord_path_prefix+'planet-'+str(i*records_per_file)+'-'+str(len(image_ids))+'.tfrecord'
                  for i in range(num_shards)]

print(tfrecord_paths)

tag_set = set()
for tag_list in tags:
    for elem in tag_list:
        tag_set.add(elem)
tag_dict = dict()
tag_index = 0
for tag in tag_set:
    tag_dict[tag] = tag_index
    tag_index = tag_index+1

tfrecord_writer = None
num_processed = 0
for i in range(num_shards):
    tfrecord_writer = tf.io.TFRecordWriter(tfrecord_paths[i])
    image_ids_shard = image_ids[i*records_per_file: (i+1)*records_per_file] if i!=num_shards-1 \
                        else image_ids[i*records_per_file: len(image_ids)]
    tags_shard = tags[i*records_per_file:(i+1)*records_per_file] if i!=num_shards-1 \
                    else tags[i*records_per_file:len(image_ids)]
    for image_id, image_tags in zip(image_ids_shard, tags_shard):
        image_path = images_path_prefix + image_id + '.tif'
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        one_hot_indices = [tag_dict[tag] for tag in image_tags]
        one_hot = np.asarray([(1 if i in one_hot_indices else 0) for i in range(len(tag_dict))], dtype=np.uint8)
        #encode
        image_string = image.tostring()
        one_hot_string = one_hot.tostring()
        feature_dict = {'image_string':_bytes_feature(image_string),
                       'one_hot_string':_bytes_feature(one_hot_string)}
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        tfrecord_writer.write(example.SerializeToString())
    num_processed = num_processed + len(tags_shard)
    print('Processed: '+str(num_processed)+ ' files.')

tfrecord_writer.close()