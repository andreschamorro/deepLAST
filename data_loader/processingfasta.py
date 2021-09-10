import os
import re
import time
import json
import gzip
import random
import argparse
import multiprocessing as mp
import numpy as np
import itertools
import tqdm
from functools import partial, reduce
from utils.dna2vec import MultiKModel

from Bio import SeqIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def get_uncompressed_size(filename):
    with gzip.open(filename, 'rb') as fd:
        fd.seek(0, 2)
        size = fd.tell()
    return size

# Utilities serialize data into a TFRecord
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def nearby(seq):
    sv = seq.replace('N', '')
    if len(sv) > 2:
        yield sv
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
            vecs.append(np.average(sub_v, axis=0) if len(sub_v) > 0 else np.zeros(model.vec_dim))
    return np.array(vecs)

class TFRecordShards(object):
    def __init__(self, filebase, num_shards=4):
        self.filebase = filebase
        self.num_shards = num_shards
        self.train_writers = [
                tf.io.TFRecordWriter("%s_train_%02i-of-%02i.tfrecord" % (self.filebase, i, self.num_shards))
                for i in range(self.num_shards)
                ]
        self.val_writers = [
                tf.io.TFRecordWriter("%s_val_%02i-of-%02i.tfrecord" % (self.filebase, i, self.num_shards))
                for i in range(self.num_shards)
                ]
        self.test_writers = [
                tf.io.TFRecordWriter("%s_test_%02i-of-%02i.tfrecord" % (self.filebase, i, self.num_shards))
                for i in range(self.num_shards)
                ]

        self.info = [
                {
                    'train_len': 0,
                    'validation_len': 0,
                    'test_len': 0}
                for i in range(self.num_shards)
                ]

    def _pick_output_shard(self):
        return random.randint(0, self.num_shards-1)

    def _pick_output_part(self):
        return random.random()

    def close(self):
        [w.close() for w in self.train_writers]
        [w.close() for w in self.val_writers]
        [w.close() for w in self.test_writers]

    def write_info(self):
        for i, info in enumerate(self.info):
                with open("%s_info_%02i-of-%02i.json" % (self.filebase, i, self.num_shards), 'w') as outfile:
                    json.dump(info, outfile)

    def write(self, ex):
        rp = self._pick_output_part()
        sp = self._pick_output_shard()
        if rp < 0.80:
            self.train_writers[sp].write(ex.SerializeToString())
            self.info[sp]['train_len'] += 1
        elif rp < 0.90:
            self.val_writers[sp].write(ex.SerializeToString())
            self.info[sp]['validation_len'] += 1
        else:
            self.test_writers[sp].write(ex.SerializeToString())
            self.info[sp]['test_len'] += 1

class Scribe(mp.Process):

    def __init__(self, listen_queue, filebase):
        mp.Process.__init__(self)
        self.listen_queue = listen_queue
        self.tfrecordshadrs = TFRecordShards(filebase)

    def run(self):
        proc_name = self.name
        while True:
            next_sentence = self.listen_queue.get()
            if next_sentence is None:
                # Poison pill means shutdown
                print('Scribe {}: finish'.format(proc_name))
                self.listen_queue.task_done()
                self.tfrecordshadrs.write_info()
                self.tfrecordshadrs.close()
                break
            self.tfrecordshadrs.write(next_sentence)
            self.listen_queue.task_done()

class Reader(mp.Process):

    def __init__(self, book_queue, listen_queue, classes_list, ignore_dict, pos, d2v_model, k_size):
        mp.Process.__init__(self)
        self.book_queue = book_queue
        self.listen_queue = listen_queue
        self.classes_list = classes_list
        self.ignore_dict = ignore_dict
        self.pos = pos
        self.d2v_model = d2v_model
        self.k_size = k_size

    @staticmethod
    def _make_tfexample(sid, seq_array, label):
        features = {
                'id': _bytes_feature(sid.encode('utf-8')),
                'shape_0': _int64_feature(seq_array.shape[0]),
                'shape_1': _int64_feature(seq_array.shape[1]),
                'sequence': _floats_feature(seq_array.flatten().tolist()),
                'label': _int64_feature(label)
                }
        return tf.train.Example(features=tf.train.Features(feature=features))

    def run(self):
        proc_name = self.name
        while True:
            next_book = self.book_queue.get()
            if next_book is None:
                # Poison pill means shutdown
                print('Reader {}: finish'.format(proc_name))
                self.book_queue.task_done()
                break
            with tqdm.tqdm(total=get_uncompressed_size(next_book), desc=os.path.basename(next_book), position=self.pos) as pbar:
                cls = os.path.dirname(next_book).split('/')[-2]
                label = self.classes_list.index(cls)
                with gzip.open(next_book, 'rt') as fasta:
                    for record in SeqIO.parse(fasta, "fastq"):
                        pbar.update(2*len(record))
                        if (record.id in self.ignore_dict and self.ignore_dict[record.id] == label):
                            continue
                        ex = Reader._make_tfexample(record.id, dna2vec(str(record.seq), self.d2v_model, self.k_size), label)
                        self.listen_queue.put(ex)
            self.book_queue.task_done()

def run(fafiles, classes, out_dir, d2v_model_file=None, k_size=8, ignore_file=None):
    """ main function """

    fafiles = random.sample(fafiles, k=len(fafiles))
    ignore_d = {}

    try:
        d2v_model = MultiKModel(d2v_model_file)
    except FileNotFoundError:
        print("d2v model file not found")
        return

    if ignore_file is not None:
        with open(ignore_file) as igfile:
            for line in igfile:
                (cls, key) = line.strip().split(',')
                ignore_d[key] = classes.index(cls)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        # directory already exists
        pass

    num_scribes = 8
    num_readers = 16

    start_time = time.time()
    with mp.Manager() as manager:
        ignore_dict = manager.dict(ignore_d)
        classes_list = manager.list(classes)
        book_queue = manager.JoinableQueue()
        listen_queue = manager.JoinableQueue()
        # Scribes work first
        print('Creating {} scribes'.format(num_scribes))
        scribes = [
                Scribe(listen_queue, out_dir+"/l%02i"%(i))
                for i in range(num_scribes)
                ]

        for s in scribes:
            s.start()

        # Readers work second
        print('Creating {} readers'.format(num_readers))
        readers = [
                Reader(book_queue, listen_queue, classes_list, ignore_dict, i, d2v_model, k_size)
                for i in range(num_readers)
                ]

        for r in readers:
            r.start()

        # Enqueue files 
        for fafile in fafiles:
            book_queue.put(fafile)

        # Add a poison pill for each readers 
        for i in range(num_readers):
            book_queue.put(None)

        # Wait for all of the readers to finish
        book_queue.join()

        # Add a poison pill for each scribes 
        for i in range(num_scribes):
            listen_queue.put(None)

        # Wait for all of the scribes to finish
        listen_queue.join()

    print(f"Processing TFExamples: --- {(time.time() - start_time)} seconds ---")
