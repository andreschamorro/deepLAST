from functools import partial
import numpy as np
import tensorflow as tf
from . import dna2bit
from configs.config import Options

def nearby(seq):
    yield seq.replace('N', '')
    for sv in seq.split('N'):
        if len(sv) > 2:
            yield sv

def dna2vec(dnalist, model, k):
    vecs = []
    for i in range(len(dnalist)-k+1):
        sub = ''.join(dnalist[i:i+k])
        try:
            vecs.append(model.vector(sub))
        except KeyError:
            sub_v = [model.vector(sv) for sv in nearby(sub)]
            return vecs.append(np.average(sub_v, axis=0) if len(sub_v) > 0 else np.zeros(100))
    return vecs

def bytes2vec(byts, model, k):
    byts = byts.numpy()
    dnaSize = int.from_bytes(byts[:4], byteorder='big')
    # Unpack blocks of Ns in the file
    blockCount, blockStarts, blockSizes = dna2bit.unpack_n_blocks(byts[4:])
    # DNA unpacked to nucleotide list
    dnalist = dna2bit.unpack_dna(byts[8*(blockCount+1):])[:dnaSize]
    rcolist = dna2bit.unpack_rco(byts[8*(blockCount+1):])[-dnaSize:]
    # Change unknown nucleotide in sequence
    for st, bs in zip(blockStarts, blockSizes):
        dnalist[st:st+bs] = bs*['N']
        rcolist[-(st+bs):-st] = bs*['N']

    return [[dna2vec(dnalist, model, k=k), dna2vec(rcolist, model, k=k)]]


def get_dataset(model, options: Options):
    """Reads in and processes the TFRecords dataset.
    Buids a pipeline that returns pairs of features, label.
    """
    
    # Define field names, types, and sizes for TFRecords.
    features = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    
    def _process_input(proto_seq):
        """Helper function for input function that parses a serialized example."""
        parsed_features = tf.io.parse_single_example(
            serialized=proto_seq, features=features)

        b2v = partial(bytes2vec, model=model, k=options.k_size)
        seqfets = tf.py_function(func=b2v, inp=[parsed_features['sequence']], Tout=tf.float32)        
        return seqfets, parsed_features['label']
    
    dta_pattern = options.data_dir + "/l??_train_??-of-??.tfrecord"
    # First just list all file pathes to the sharded tfrecord dataset.
    dta = tf.data.TFRecordDataset.list_files(dta_pattern)
    # Make sure to fully shuffle the list of tfrecord files.
    dta = dta.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    # Preprocesses 16 files concurrently and interleaves records from each file into a single, unified dataset.
    dta = dta.interleave(
        tf.data.TFRecordDataset, cycle_length=16, block_length=1)
    dta = dta.map(map_func=_process_input, num_parallel_calls=8)
    dta = dta.batch(batch_size=options.batch_size).repeat(count=options.num_epochs)
    
    dva_pattern = options.data_dir + "/l??_val_??-of-??.tfrecord"
    # First just list all file pathes to the sharded tfrecord dataset.
    dva = tf.data.TFRecordDataset.list_files(dva_pattern)
    # Make sure to fully shuffle the list of tfrecord files.
    dva = dva.shuffle(buffer_size=1000)
    # Preprocesses 16 files concurrently and interleaves records from each file into a single, unified dataset.
    dva = dva.interleave(
        tf.data.TFRecordDataset, cycle_length=16, block_length=1)
    dva = dva.map(map_func=_process_input, num_parallel_calls=8)
    dva = dva.batch(batch_size=options.batch_size)
    
    dte_pattern = options.data_dir + "/l??_test_??-of-??.tfrecord"
    # First just list all file pathes to the sharded tfrecord dataset.
    dte = tf.data.TFRecordDataset.list_files(dva_pattern)
    # Make sure to fully shuffle the list of tfrecord files.
    dte = dte.shuffle(buffer_size=1000)
    # Preprocesses 16 files concurrently and interleaves records from each file into a single, unified dataset.
    dte = dte.interleave(
        tf.data.TFRecordDataset, cycle_length=16, block_length=1)
    dte = dte.map(map_func=_process_input, num_parallel_calls=8)
    dte = dte.batch(batch_size=options.batch_size)
    
    return dta, dva, dte
