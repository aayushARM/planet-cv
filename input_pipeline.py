
import tensorflow as tf

# Enable eager mode for easy debugging.
tf.enable_eager_execution()
train_samples = int(0.8*40479)
feature_dict = {'image_string': tf.FixedLenFeature([], tf.string),
                'one_hot_string': tf.FixedLenFeature([], tf.string)}


def parse_and_augment(proto_record):
    features = tf.parse_single_example(proto_record, feature_dict)
    # already normalized
    image_flat = tf.decode_raw(features['image_string'], tf.float32)
    image = tf.reshape(image_flat, [256, 256, 4])
    # Flipping satellite chips/images in any of the four directions doesn't change the labels,
    # and also provides good data augmentation. Note that by default both functions below flip 
    # with a probability=0.5.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    one_hot = tf.decode_raw(features['one_hot_string'], tf.uint8)

    return image, one_hot


def get_data(files, batch_size, num_shards, shuffle_buffer_size):
    raw_data = files.interleave(tf.data.TFRecordDataset, cycle_length=num_shards)
    parsed_data = raw_data.map(map_func=parse_and_augment, num_parallel_calls=num_shards)

    train_data = parsed_data.take(train_samples)
    val_data = parsed_data.skip(train_samples)

    train_data = train_data.repeat().batch(batch_size)
    #train_data = train_data.shuffle(shuffle_buffer_size).repeat().batch(batch_size)

    val_data = val_data.repeat().batch(batch_size)

    train_data = train_data.prefetch(1)
    val_data = val_data.prefetch(1)
    
    # For debugging:
    # count = 0
    # for (_, one_hot1), (_, one_hot2) in zip(train_data, val_data):
    #     print(repr(one_hot1))
    #     print(repr(one_hot2))
    #     count = count + 1
    #     print(count)
    #
    return train_data, val_data

def parse_test(proto_record):
    features = tf.parse_single_example(proto_record, feature_dict)
    # already normalized
    image_flat = tf.decode_raw(features['image_string'], tf.float32)
    image = tf.reshape(image_flat, [256, 256, 4])

    return image


def get_data_test(files, batch_size, num_shards):
    raw_images = tf.data.TFRecordDataset(files)
    parsed_images = raw_images.map(map_func=parse_test, num_parallel_calls=num_shards)
    parsed_images = parsed_images.batch(batch_size)
    parsed_images.prefetch(1)
    return parsed_images
