# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import numpy as np
from typing import Dict, Any
from datetime import datetime
from configs.config import Options
from tensorboard.plugins.hparams.api import KerasCallback
from hyperopt import fmin, tpe, hp, Trials, STATUS_FAIL, STATUS_OK

def _update_options(options: Options, dictionary: Dict[str, Any]) -> Options:
    for key, value in dictionary.items():
        options[key] = value
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

def train_hvd(options: Options, d2v_model, checkpoint_dir, extra_callback=None, cuda=False):
    import tensorflow as tf
    import horovod.tensorflow.keras as hvd
    from data_loader.data_reader import get_dataset
    from models.models import deepLAST
    
    # Horovod: initialize Horovod.
    hvd.init()
    
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    if cuda:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dtrain, dval, dtest = get_dataset(d2v_model, options)
    for x, y in dtrain:
        print("READ DATA")
        print(x.shape)
        break
    last_model = deepLAST.get_model(options)

    # Horovod: adjust learning rate based on number of GPUs.
    options.learning_rate = options.learning_rate * hvd.size()
    opt = deepLAST.get_optimizer(options)
    
    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(
        opt, backward_passes_per_step=1, average_aggregated_gradients=True)
    
    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    last_model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                        optimizer=opt,
                        metrics=['accuracy'],
                        experimental_run_tf_function=False)
    
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    
        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
    
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=options.learning_rate, warmup_epochs=3, verbose=1),
    ]

    
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
#        param_str = 'learning_rate_{lr}_batch_size_{bs}'.format(lr=learning_rate, bs=batch_size)
#        checkpoint_dir_for_this_trial = os.path.join(checkpoint_dir, param_str)
#        local_ckpt_path = os.path.join(checkpoint_dir_for_this_trial, 'checkpoint-{epoch}.ckpt')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, '{epoch:002d}'),
                monitor='val_loss',
                verbose=0,
                mode='min',
                save_freq='epoch',
                save_best_only=True,
                save_weights_only=True)
        callbacks.append(model_checkpoint_callback)
        if extra_callback: callbacks.append(extra_callback)
    
    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0
    
    # Train the model.
    # Horovod: adjust number of steps based on number of GPUs.
    last_model.fit(dtrain,
                   epochs=options.num_epochs,
                   callbacks=callbacks,
                   validation_data=dval, validation_steps=1,
                   validation_batch_size=options.batch_size, validation_freq=1,
                   steps_per_epoch=options.steps_per_epoch//hvd.size(),
                   verbose=verbose,
                   shuffle=True)

    return last_model.evaluate(dtest, steps=options.evaluation_steps)

# Objetive function
def build_and_train(d2v_model, options: Options, options_dict):
    """
    An example train method that calls into HorovodRunner.
    This method is passed to hyperopt.fmin().
    
    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    options = _update_options(options, options_dict)
    checkpoint_dir = _create_check_dir(options)

    results = {
            'loss': np.inf,
            'Metrics': None,
            'options': options.todict(),
            'checkpoint_dir': None,
            'status': STATUS_FAIL,
            'error': "",
    }

    extra_callback = [KerasCallback(checkpoint_dir, options)]
    try:
        loss, acc = train_hvd(
                options,
                d2v_model,
                checkpoint_dir,
                extra_callback=extra_callback)
    except Exception as err:  # pylint: disable=broad-except
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno, err)

        results["error"] = str(err)
        results["status"] = STATUS_FAIL
    else:
        results["checkpoint_dir"] = checkpoint_dir
        results["status"] = STATUS_OK
        results["loss"] = loss
        results["Metrics"] = acc
        if np.isnan(results["loss"]):
            results["status"] = STATUS_FAIL
            results["loss"] = np.inf
    return results

def run(d2v_model=None):
    import os
    import numpy as np
    import pickle
    from functools import partial
    from pkg_resources import resource_filename
    from utils.dna2vec import MultiKModel

    search_space = {
        'gru_units': hp.qnormal('gru_units', 34, 5, 2),
        'gru_dropout': hp.uniform('gru_dropout', 0, 0.4),
        'momentum': hp.uniform('momentum', 0, 1),
        'rho': hp.uniform('decay', 0, 1),
        'repeat_probability': hp.uniform('repeat_probability', 0, 0.49),
        'learning_rate': hp.lognormal('learning_rate', -7, 0.5),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
    }

    last_rnn_options = resource_filename(
            'configs',
            'last_rnn.json'
    )
    options = Options.get_options_from_json(last_rnn_options)
    results_path= os.path.join(options.summary_dir, "tf_results.pkl")
    try:
        with open(results_path, "rb") as file:
            trials = pickle.load(file)
    except FileNotFoundError:
        trials = Trials()
#        _LOGGER.info("Starting from scratch: new trials.")
    else:
#        _LOGGER.warning("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
#        _LOGGER.info("Rerunning from %d trials to add another one.",
#                     len(trials.trials))
    if not d2v_model:
        d2v_pretrained = resource_filename(
            'models',
            os.path.join('pretrained', 'dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'))
        d2v_model = MultiKModel(d2v_pretrained)
    objetive = partial(build_and_train, d2v_model, options)
    algo=tpe.suggest

    best_param = fmin(
            fn=objetive,
            space=search_space,
            algo=algo,
            max_evals=8,
            trials=trials
    )

    with open(results_path, "wb") as file:
        pickle.dump(trials, file)
