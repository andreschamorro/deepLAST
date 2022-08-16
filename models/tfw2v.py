import os
from datetime import datetime
from pkg_resources import resource_filename
from utils import logger as Logger
import time  # To time our operations

from configs.config import Options
from data_loader.tfkmer import KmerTokenizer, Kmer2VecDatasetBuilder
from models.models import Word2VecModel
#, get_train_step_signature

import tensorflow as tf

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

def build_dataset(options: Options):
    tokenizer = KmerTokenizer(
            k=options.k, max_vocab_size=options.max_vocab_size, min_count=options.min_count, sample=options.sample)
    tokenizer.build_vocab(options.features_files)

    builder = Kmer2VecDatasetBuilder(tokenizer,
            arch=options.arch,
            algm=options.algm,
            epochs=options.num_epochs,
            batch_size=options.batch_size,
            window_size=options.window_size)
    return tokenizer, builder, builder.build_dataset(options.features_files)

def build_model(options: Options, tokenizer, builder, logger, prev_checkpoint=None, continue_train=True):
    word2vec = Word2VecModel(tokenizer.unigram_counts,
            arch=options.arch,
            algm=options.algm,
            hidden_size=options.hidden_size,
            batch_size=options.batch_size,
            window_size=options.window_size,
            max_depth=builder._max_depth,
            negatives=options.negatives,
            power=options.power,
            alpha=options.alpha,
            min_alpha=options.min_alpha,
            add_bias=options.add_bias)

    word2vec.compile(optimizer=tf.keras.optimizers.SGD(1.0))
    return word2vec, True

def training(options: Options, model, dataset, builder, checkpoint_dir, logger, extra_callback=None):

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)
    callbacks = [model_checkpoint_callback]
    if extra_callback: callbacks.append(extra_callback)

    logger.info("Transvec training...")
    start_time = time.time()
    model.fit(dataset, epochs=options.num_epochs, callbacks=callbacks)
    stop_time = time.time()
    logger.info('Time to train the model: {} mins'.format(round((stop_time - start_time) / 60, 2)))

    return model

def run(prev_checkpoint=None, continue_train=True, save_vocab=False, save_model=True):
    fast_options = resource_filename(
            'configs',
            'tf_transvec.json'
    )
    options = Options.get_options_from_json(fast_options)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(options.logs_dir, 'logs-{}'.format(now))
    logger = Logger.get_logger('transvec', log_dir)

    logger.info("Create checkpoint dir")
    checkpoint_dir = _create_check_dir(options)

    logger.info("Build dataset")
    tokenizer, builder, dataset = build_dataset(options)
    logger.info("Build model")
    model, update = build_model(options, tokenizer, builder, logger, prev_checkpoint, continue_train)

    logger.info("Training...")
    model = training(options, model, dataset, builder, checkpoint_dir, logger)

    if save_model:
        syn0_final = model.weights[0].numpy()
        np.save(os.path.join(checkpoint_dir, 'syn0_final'), syn0_final)
        with tf.io.gfile.GFile(os.path.join(checkpoint_dir, 'vocab.txt'), 'w') as f:
            for w in tokenizer.table_words:
                f.write(w + '\n')
        logger.info('Word embeddings saved to', 
                os.path.join(checkpoint_dir, 'syn0_final.npy'))
        logger.info('Vocabulary saved to', os.path.join(checkpoint_dir, 'vocab.txt'))
