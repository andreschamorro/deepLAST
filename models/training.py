import os
import sys
import json
import argparse
import time
import pickle
from typing import Dict, Any
from datetime import datetime
from pkg_resources import resource_filename

from utils.dna2vec import MultiKModel

from configs.config import Options
from utils import logger as Logger

def _create_check_dir(options) -> str:
    """Standarized formating of checkpoint dirs.
    Args:
        options (Options): information about the projects name.
    Returns:
        str: standarized logdir path.
    """
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join(options.checkpoint_dir, "run_training-{}".format(now))
    return checkpoint_dir

def training(options: Options, d2v_model, checkpoint_dir, logger, extra_callback=None):
    from data_loader.data_reader import get_dataset_training
    from models.models import deepLAST
    import json
    import tensorflow as tf

    logger.info("Load dataset")
    dtrain, dval = get_dataset_training(d2v_model, options)
    last_model = deepLAST.get_model(options)

    opt = deepLAST.get_optimizer(options)
    last_model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
            optimizer=opt,
            metrics=['accuracy', tf.keras.metrics.Precision(name='precission'), tf.keras.metrics.Recall(name='recall')],
            experimental_run_tf_function=False)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, '{epoch:002d}'),
            monitor='val_recall',
            verbose=0,
            mode='max',
            save_freq='epoch',
            save_best_only=True,
            save_weights_only=True)
    callbacks = [model_checkpoint_callback]
    if extra_callback: callbacks.append(extra_callback)

    # Train the model.
    logger.info("Starting model training")
    last_history = last_model.fit(dtrain, epochs=options.num_epochs,
                                   callbacks=callbacks, validation_data=dval, 
                                   validation_steps=1, validation_batch_size=options.batch_size, 
                                   validation_freq=1, steps_per_epoch=options.steps_per_epoch,
                                   verbose=1, shuffle=True)
    with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as file:
         json.dump(str(last_history.history), file)

    return last_model

def testing(options: Options, d2v_model, last_model, logger):
    from data_loader.data_reader import get_dataset_testing
    import tensorflow as tf

    logger.info("Load dataset")
    dtest = get_dataset_testing(d2v_model, options)
    logger.info("Starting model testing")
    return last_model.evaluate(dtest, steps=options.evaluation_steps, workers=32, use_multiprocessing=True)

def run(d2v_model):

    last_rnn_options = resource_filename(
            'configs',
            'last_rnn.json'
    )
    options = Options.get_options_from_json(last_rnn_options)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(options.logs_dir, 'logs-{}'.format(now))
    logger = Logger.get_logger('hyper_opt', log_dir)

    #logger.info("Load dna2vec model")
    #d2v_pretrained = resource_filename(
    #    'models',
    #    os.path.join('pretrained', 'dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'))
    #d2v_model = MultiKModel(d2v_pretrained)

    logger.info("Create checkpoint dir")
    checkpoint_dir = _create_check_dir(options)

    last_model = training(options, d2v_model, checkpoint_dir, logger)

    acc, pre, rec = testing(options, d2v_model, last_model, logger)
    logger.info("Save metrics")
    with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as file:
        json.dump({"accuracy": acc, "precission": pre, "recall": rec}, file)
