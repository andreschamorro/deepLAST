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

    #train_step_signature = get_train_step_signature(
    #        options.arch, options.algm, options.batch_size, options.window_size, builder._max_depth)
    #optimizer = tf.keras.optimizers.SGD(1.0)

    logger.info("Transvec training...")
    start_time = time.time()
    model.fit(dataset, epochs=options.num_epochs)
    stop_time = time.time()
    logger.info('Time to train the model: {} mins'.format(round((stop_time - start_time) / 60, 2)))

    #average_loss = 0.
    ## Iterate over epochs.
    #for epoch in range(options.num_epochs):
    #    print("Start of epoch %d" % (epoch,))
    #    for step, (inputs, labels, progress) in enumerate(dataset):
    #        loss, learning_rate = train_step(inputs, labels, progress)
    #        average_loss += loss.numpy().mean()
    #        if step % options.log_per_steps == 0:
    #            if step > 0:
    #                average_loss /= options.log_per_steps
    #            print('step:', step, 'average_loss:', average_loss,
    #                        'learning_rate:', learning_rate.numpy())
    #            average_loss = 0.

    #syn0_final = model.weights[0].numpy()
    #np.save(os.path.join(FLAGS.out_dir, 'syn0_final'), syn0_final)
    #with tf.io.gfile.GFile(os.path.join(FLAGS.out_dir, 'vocab.txt'), 'w') as f:
    #    for w in tokenizer.table_words:
    #        f.write(w + '\n')
    #print('Word embeddings saved to', 
    #        os.path.join(FLAGS.out_dir, 'syn0_final.npy'))
    #print('Vocabulary saved to', os.path.join(FLAGS.out_dir, 'vocab.txt'))

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
