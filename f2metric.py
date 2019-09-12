# This file consists of code related to f2 score computation. Since implemented as Tensorflow nodes, this score acts as a part 
# of training graph and is differentiable during backpropogation just like other graph nodes.

from tensorflow.python.keras import backend as K
import tensorflow as tf

def f_score(y_true, y_pred, threshold=0.1, beta=2):

    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    score = (1+beta**2) * ((precision * recall) / ((beta**2)*precision + recall))
    score = tf.where(tf.is_nan(score), tf.constant(0.0, dtype=tf.float64), score)

    return score


def tp_score(y_true, y_pred, threshold=0.1):

    tp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))

    return tp


def fp_score(y_true, y_pred, threshold=0.1):

    fp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(K.abs(y_true - K.ones_like(y_true)))), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=-1
    )

    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))

    return fp


def fn_score(y_true, y_pred, threshold=0.1):

    fn_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.abs(K.cast(K.greater(y_pred, K.constant(threshold)), 'float') - K.ones_like(y_pred)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    return fn


def precision_score(y_true, y_pred, threshold=0.1):

    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)

    return tp / (tp + fp)


def recall_score(y_true, y_pred, threshold=0.1):

    tp = tp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)

    return tp / (tp + fn)
