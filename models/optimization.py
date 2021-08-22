# Specify training parameters
epochs = 2
num_classes = 2

def train_hvd(learning_rate, batch_size, checkpoint_dir):
    """
    This function is passed to Horovod and executed on each worker.
    Pass in the hyperparameters we will tune with Hyperopt.
    """
    # Import tensorflow modules to each worker
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Sequential
    import tensorflow as tf
    from tensorflow import keras
    import horovod.tensorflow.keras as hvd
    import shutil

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    # These steps are skipped on a CPU cluster
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    (x_train, y_train), (x_test, y_test) = get_dataset(num_classes, hvd.rank(), hvd.size())
    model = get_model(num_classes)

    # Adjust learning rate based on number of GPUs
    optimizer = keras.optimizers.Adadelta(lr=learning_rate * hvd.size())

    # Use the Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer)
    model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    # Create a callback to broadcast the initial variable states from rank 0 to all other processes.
    # This is required to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.
    callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),]

    # Save checkpoints only on worker 0 to prevent conflicts between workers
    if hvd.rank() == 0:
        param_str = 'learning_rate_{lr}_batch_size_{bs}'.format(lr=learning_rate, bs=batch_size)
        checkpoint_dir_for_this_trial = os.path.join(checkpoint_dir, param_str)
        local_ckpt_path = os.path.join(checkpoint_dir_for_this_trial, 'checkpoint-{epoch}.ckpt')
        callbacks.append(keras.callbacks.ModelCheckpoint(local_ckpt_path, save_weights_only = True))

    model.fit(x_train, y_train,
            batch_size=batch_size,
            callbacks=callbacks,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))

    return model.evaluate(x_test, y_test)
