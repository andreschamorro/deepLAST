import os
from datetime import datetime
from pkg_resources import resource_filename
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec
from configs.config import Options
from data_loader.kmer_generator import KmerGenerator
from data_loader.threadedgenerator import ThreadedGenerator 
from utils import logger as Logger
import time  # To time our operations

def _create_check_dir(options) -> str:
    """Standarized formating of checkpoint dirs.
    Args:
        options (Options): information about the projects name.
    Returns:
        str: standarized logdir path.
    """
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join(options.checkpoint_dir, "transvec_training-{}".format(now))
    # create file handler which logs even debug messages
    os.makedirs(f'{checkpoint_dir}', exist_ok=True)
    return checkpoint_dir

class ModelCheckpoint(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, filepath, save_last=10):
        self.filepath = filepath
        self.epoch = 0

    def on_epoch_end(self, model):
        self._save_model(model, self.epoch, None, None)
        self.epoch += 1

    def _save_model(self, model, epoch, batch, logs):
        logs = logs or {}
        output_path = self._get_file_path(epoch % save_last, batch, logs)
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
    # make_or_restore_model
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        logger.info('Transvec restoring from {} epoch'.format(latest_checkpoint))
        return Doc2Vec.load(latest_checkpoint), True
    else:
        return Doc2Vec(vector_size=options.vector_size, workers=options.workers), False

def build_vocab(model, kmer_seq_iterable, checkpoint_dir, logger, update=False, save=False):

    # build the vocabulary
    logger.info("Build the vocabulary")
    start_time = time.time()
    model.build_vocab(corpus_iterable=kmer_seq_iterable, update=update)
    stop_time = time.time()
    if save:
        model.save(os.path.join(checkpoint_dir, 'model_vocab.model'))
    logger.info('Time to build vocab: {} mins'.format(round((stop_time - start_time) / 60, 2)))
    
    return model

def training(options: Options, model, kmer_seq_iterable, checkpoint_dir, logger, extra_callback=None):

    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, '{epoch:002d}'))
    callbacks = [model_checkpoint_callback]
    if extra_callback: callbacks.append(extra_callback)

    # train
    logger.info("Transvec training...")
    start_time = time.time()
    model.train(corpus_iterable=kmer_seq_iterable, total_examples=model.corpus_count, epochs=options.num_epochs, callbacks=callbacks)
    stop_time = time.time()
    logger.info('Time to train the model: {} mins'.format(round((stop_time - start_time) / 60, 2)))

    return model

def write_vec(model, outpath):
    out_filename = '{}.w2v'.format(outpath)
    model.wv.save_word2vec_format(out_filename, binary=False)

def run(prev_checkpoint=None, continue_train=True, save_vocab=False, save_model=True):
    fast_options = resource_filename(
            'configs',
            'transvec.json'
    )
    options = Options.get_options_from_json(fast_options)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(options.logs_dir, 'logs-{}'.format(now))
    logger = Logger.get_logger('transvec', log_dir)

    logger.info("Create checkpoint dir")
    checkpoint_dir = _create_check_dir(options)

    logger.info("Build model")
    model, update = build_model(options, logger, prev_checkpoint, continue_train)

    logger.info("Load kmer generator")
    kmer_seq_iterable = KmerGenerator(
            options.fasta_file,
            options.gff_file,
            options.k_low,
            options.k_high,
            options.rseed_trainset,
            logger=logger)

    thread_generator = ThreadedGenerator(kmer_seq_iterable, queue_maxsize=4096, daemon=True)

    model = build_vocab(model, thread_generator, checkpoint_dir, logger, update=update, save=save_vocab)

    model = training(options, model, kmer_seq_iterable, checkpoint_dir, logger)
    if save_model:
        logger.info("Save wv vectors")
        write_vec(model, os.path.join(checkpoint_dir, 'transvec_model'))
