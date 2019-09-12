
import tensorflow as tf
import tensorflow.python.keras as tfk
import input_pipeline
import seresnet
from f2metric import f_score

def main():
    # Some hyper params, tuned multiple times as needed.
    batch_size = 24
    total_samples = 40479
    train_batches = int(total_samples * 0.8 / batch_size)
    val_batches = int(total_samples * 0.2 / batch_size)
    num_shards = 4
    shuffle_buffer_size = 6000
    learning_rate = 0.0001
    # Change below paths as needed.
    checkpoint_path = '/media/aayush/Other/Planet/SEResnet154_ckpts/weights'
    files = tf.data.Dataset.list_files('/media/aayush/Other/Planet/tfrecords/train/*.tfrecord')
    # TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

    train_data, val_data = input_pipeline.get_data(files, batch_size, num_shards, shuffle_buffer_size)

    model = seresnet.SEResNet154(input_shape=(256, 256, 4), classes=17)
    #print(model.summary())
    
    # Note that we're using a custom metric(f2-score) during model compilation.
    model.compile(optimizer=tfk.optimizers.Adam(lr=learning_rate), loss='mean_squared_error',
                  metrics=[f_score])
    
    # Similar as  above, we're monitoring a custom f2-score for validation in the Checkpoint callback.
    ckpt_callback = tfk.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1,
                                                  monitor='val_f_score',
                                                  save_best_only=True, mode='max')

    tensorboard_callback = tfk.callbacks.TensorBoard(
        log_dir='/media/aayush/Other/Planet/SEResNet154_logs',
        histogram_freq=0, batch_size=batch_size, write_graph=True,
        write_grads=True, update_freq='batch')

    model.load_weights(checkpoint_path)

    # tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    #     model, strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

    model.fit(train_data, validation_data=val_data, epochs=50, verbose=1, steps_per_epoch=10, validation_steps=10,
              callbacks=[tensorboard_callback, ckpt_callback], initial_epoch=0, validation_freq=1)

    #model.evaluate(val_data, steps=50, verbose=2, workers=4, use_multiprocessing=True)

main()
