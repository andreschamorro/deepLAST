import os
import sys
import json
import traceback
import argparse
import time
import numpy as np
import pickle
from typing import Dict, Any
from datetime import datetime
from functools import partial
import portpicker
import multiprocessing
from pkg_resources import resource_filename

from utils.dna2vec import MultiKModel

from configs.config import Options
from utils import logger as Logger
from models.callbacks import create_callbacks

from hyperopt import fmin, tpe, hp, Trials, STATUS_FAIL, STATUS_OK

def _update_options(options: Options, dictionary: Dict[str, Any]) -> Options:
    for key, value in dictionary.items():
        options[key] = value
    options.units = int(options.units)
    return options

def _create_check_dir(options) -> str:
    """Standarized formating of checkpoint dirs.
    Args:
        options (Options): information about the projects name.
    Returns:
        str: standarized logdir path.
    """
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join(options.checkpoint_dir, "run-{}".format(now))
    return checkpoint_dir


def train_and_eval(options: Options, d2v_model, checkpoint_dir, logger, extra_callback=None):
    from data_loader.data_reader import get_dataset
    from models.models import deepLAST
    import json
    import tensorflow as tf

    dtrain, dval, dtest = get_dataset(d2v_model, options)
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

    logger.info("Starting model evaluation")
## DEBUG
## steps=options.evaluation_steps
    return last_model.evaluate(dtest, steps=32, workers=32, use_multiprocessing=True)

# Objetive function
def build_and_train(d2v_model, logger, options: Options, options_dict):
    """
    An example train method that calls into HorovodRunner.
    This method is passed to hyperopt.fmin().
    
    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    options = _update_options(options, options_dict)
## DEBUG
    options.data_dir = '../' + options.data_dir
    checkpoint_dir = _create_check_dir(options)

    results = {
            'loss': np.inf,
            'Metrics': None,
            'options': options.todict(),
            'checkpoint_dir': None,
            'status': STATUS_FAIL,
            'error': "",
    }

    try:
        loss, acc, precission, recall = train_and_eval(
                options,
                d2v_model,
                checkpoint_dir,
                logger)
    except Exception as err:  # pylint: disable=broad-except
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno, err)

        results["error"] = str(err)
        results["status"] = STATUS_FAIL
    else:
        results["checkpoint_dir"] = checkpoint_dir
        results["status"] = STATUS_OK
        results["loss"] = 1-recall
        results["Metrics"] = {'Loss': loss, 'Accuracy': acc, 
                'Precision': precission, 'Recall': recall}
        if np.isnan(results["loss"]):
            results["status"] = STATUS_FAIL
            results["loss"] = np.inf
    return results

def run(d2v_model=None):

    last_rnn_options = resource_filename(
            'configs',
            'last_rnn.json'
    )
    options = Options.get_options_from_json(last_rnn_options)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(options.logs_dir, 'logs-{}'.format(now))
    logger = Logger.get_logger('hyper_opt', log_dir)

    max_evals = 8
    results_path= os.path.join(options.summary_dir, "tf_results.pkl")
    try:
        with open(results_path, "rb") as file:
            trials = pickle.load(file)
    except FileNotFoundError:
        trials = Trials()
        logger.info("Starting from scratch: new trials.")
    else:
        logger.warning("Found saved Trials! Loading...")
        max_evals += len(trials.trials)
        logger.info("Rerunning from %d trials to add another one.",
                     len(trials.trials))
    if not d2v_model:
        logger.info("Load dna2vec model")
        d2v_pretrained = resource_filename(
            'models',
            os.path.join('pretrained', 'dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'))
        d2v_model = MultiKModel(d2v_pretrained)

    logger.info("Initializing hyperopt")
    logger.info("Initializing search space")
    search_space = {
        'units': hp.qnormal('units', 34, 5, 2),
        'dropout': hp.uniform('dropout', 0, 0.4),
        'momentum': hp.uniform('momentum', 0, 1),
        'rho': hp.uniform('decay', 0, 1),
        'learning_rate': hp.lognormal('learning_rate', -7, 0.5),
    }

    logger.info("Initializing objetive and search algorithms")
    objetive = partial(build_and_train, d2v_model, logger, options)
    algo=tpe.suggest

    logger.info("Starting hyperparameter tuning")
    best_param = fmin(
            fn=objetive,
            space=search_space,
            algo=algo,
            max_evals=max_evals,
            trials=trials
    )

    with open(results_path, "wb") as file:
        pickle.dump(trials, file)
    
    # Delete the `TF_CONFIG`, and kill any background tasks so they don't affect the next section.
    os.environ.pop('TF_CONFIG', None)
