import os
from datetime import datetime
import numpy as np
from pkg_resources import resource_filename
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec
from configs.config import Options
from data_loader.trns_generator import TaggedKPairRead
from data_loader.threadedgenerator import ThreadedGenerator 
from utils import logger as Logger
import time  # To time our operations

def _create_work_dir(options) -> str:
    """Standarized formating of checkpoint dirs.
    Args:
        options (Options): information about the projects name.
    Returns:
        str: standarized logdir path.
    """
    summary_dir = os.makedirs(options.summary_dir, exist_ok=True)
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join(options.checkpoint_dir, "transvec_training-{}".format(now))
    # create file handler which logs even debug messages
    os.makedirs(f'{checkpoint_dir}', exist_ok=True)
    return summary_dir, checkpoint_dir

class ModelCheckpoint(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, filepath, save_last=10):
        self.filepath = filepath
        self.save_last = save_last
        self.epoch = 0

    def on_epoch_end(self, model):
        self._save_model(model, self.epoch, None, None)
        self.epoch += 1

    def _save_model(self, model, epoch, batch, logs):
        logs = logs or {}
        output_path = self._get_file_path(epoch % self.save_last, batch, logs)
        model.save(output_path)

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
          # `filepath` may contain placeholders such as `{epoch:02d}`,`{batch:02d}`
          # and `{mape:.2f}`. A mismatch between logged metrics and the path's
          # placeholders can cause formatting to fail.
          if batch is None or 'batch' in logs:
            file_path = self.filepath.format(epoch=epoch + 1, **logs)
          else:
            file_path = self.filepath.format(
                epoch=epoch + 1, batch=batch + 1, **logs)
        except KeyError as e:
          raise KeyError(
              f'Failed to format this callback filepath: "{self.filepath}". '
              f'Reason: {e}')
        self._write_filepath = file_path 
        return self._write_filepath

class MonitorCallback(CallbackAny2Vec):
    '''Report Recall@K metric at end of each epoch

    Computes and reports Recall@K on a validation set with
    a given value of k (number of recommendations to generate).
    '''
    def __init__(self, validation, k=10, ray=False):
        self.epoch = 0
        self.validation = validation
        self.k = k
        self.ray = ray

    def on_epoch_end(self, model):
        # compute the metric we care about on a recommendation task
			  # with the validation set using the model's embedding vectors
        score = 0
        for query_item, ground_truth in self.validation:
            try:
                # get the k most similar items to the query item
                neighbors = model.dv.most_similar([model.infer_vector(query_item)], topn=self.k)
            except KeyError:
                pass
            else:
                recommendations = [item for item, distance in neighbors]
                if ground_truth in recommendations:
                    score += 1
        score /= len(self.validation)

        if self.ray:
            tune.report(recall_at_k = score)
        else:
            print(f"Epoch {self.epoch} -- Recall@{self.k}: {score}")
        self.epoch += 1

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def build_model(options: Options, logger, prev_checkpoint=None, continue_train=True):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [os.path.join(prev_checkpoint, f)
            for f in os.listdir(prev_checkpoint) if f[:2].isdigit()]
    old_model = [os.path.join(prev_checkpoint, f)
            for f in os.listdir(prev_checkpoint) if os.path.splitext(f)[1] == '.model']
    # make_or_restore_model
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        logger.info('Transvec restoring from {} epoch'.format(latest_checkpoint))
        return Doc2Vec.load(latest_checkpoint), False
    elif old_model:
        latest_model = max(old_model, key=os.path.getctime)
        logger.info('Transvec restoring from old model {}'.format(latest_model))
        return Doc2Vec.load(latest_model), False
    else:
        return Doc2Vec(vector_size=options.vector_size, dm_mean=options.dm_mean, dm=options.dm,
                dbow_words=options.dbow_words, dm_concat=options.dm_concat, 
                window=options.window_size, alpha=options.alpha, min_alpha=options.min_alpha,
                seed=options.seed, min_count=options.min_count, max_vocab_size=options.max_vocab_size,
                sample=options.sample, workers=options.workers, epochs=options.num_epochs,
                hs=options.hs, negative=options.negative, ns_exponent=options.ns_exponent,
                compute_loss=True), True

def build_vocab(model, corpus_iterable, checkpoint_dir, logger, update=False, save=False):

    # build the vocabulary
    logger.info("Build the vocabulary")
    start_time = time.time()
    model.build_vocab(corpus_iterable=corpus_iterable)
    stop_time = time.time()
    if save:
        np.save(os.path.join(checkpoint_dir, 'vocab'), model.wv.index_to_key)
    logger.info('Time to build vocab: {} mins'.format(round((stop_time - start_time) / 60, 2)))
    
    return model

def training(options: Options, model, corpus_iterable, validation_iterable, checkpoint_dir, logger, extra_callback=None):

    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, '{epoch:002d}'), save_last=2)
    model_monitor_callback = MonitorCallback(validation_iterable)
    callbacks = [model_checkpoint_callback, model_monitor_callback]
    if extra_callback: callbacks.append(extra_callback)

    # train
    logger.info("Transvec training...")
    start_time = time.time()
    model.train(corpus_iterable=corpus_iterable, 
            total_examples=model.corpus_count, total_words=model.corpus_total_words, 
            epochs=options.num_epochs, queue_factor=options.queue_factor, callbacks=callbacks)
    stop_time = time.time()
    logger.info('Time to train the model: {} mins'.format(round((stop_time - start_time) / 60, 2)))

    return model

def cipher(i):
    ids = i.split('|')
    if len(ids) > 5:
        return ids[0] + '_' + ids[5]
    else:
        return 'L1_' + i

def run(prev_checkpoint=None, continue_train=True, corpus_iterable=None, save_vocab=False, save_model=True):
    fast_options = resource_filename(
            'configs',
            'transvec.json'
    )
    options = Options.get_options_from_json(fast_options)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(options.logs_dir, 'logs-{}'.format(now))
    logger = Logger.get_logger('transvec', log_dir)

    logger.info("Create checkpoint dir")
    summary_dir, checkpoint_dir = _create_work_dir(options)

    logger.info("Create corpus file from generator")
    corpus_iterable = TaggedKPairRead(options.features_files, options.k, cipher=cipher).prefetch().shuffle()
    validation_iterable = corpus_iterable.sample(options.validation_size) 

    logger.info("Build model")
    model, update = build_model(options, logger, prev_checkpoint, continue_train)

    if update:
        model = build_vocab(model, corpus_iterable, checkpoint_dir, logger, update=update, save=save_vocab)

    model = training(options, model, corpus_iterable, validation_iterable, checkpoint_dir, logger)
    if save_model:
        logger.info("Save wv vectors")
        model.save(os.path.join(checkpoint_dir, 'transvec_model'))
