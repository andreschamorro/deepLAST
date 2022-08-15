import os
import resource
import argparse
import itertools
import re
import six
import gzip
from threading import Thread, Lock
import numpy as np
from enum import Enum
from typing import Dict, List
import gffutils
from pyfaidx import Fasta
import sqlite3
from gensim.models.doc2vec import TaggedDocument
from typing import NamedTuple
from Bio import bgzf, SeqIO
from tqdm import tqdm
import logging

class Feature(NamedTuple):
    seq: str
    sid: str

def memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E6

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

class Genome:
    def __init__(self, genome_dir, genome_file, gff_file, logger=None):
        self.genome_dir = genome_dir
        self.genome_file = genome_file
        genome_basename = os.path.splitext(os.path.basename(genome_file))
        self.feature_file = os.path.join(genome_dir, genome_basename[0]+'.fe'+ genome_basename[1])
        self.gff_file = gff_file
        self.features = None
        self.logger = logger
        self.lock = Lock()
        self.gff_db = None 

    def build_db(self, rebuild=False, build_db=True):
        self.db_name = os.path.splitext(self.gff_file)[0] + '.db'
        with self.lock:  # lock around index generation so only one thread calls method
            try:
                if os.path.exists(self.db_name) and os.path.getmtime(
                        self.db_name) >= os.path.getmtime(self.gff_file):
                    gff_db = sqlite3.connect(self.db_name, check_same_thread=False)
                    gff_db = gffutils.FeatureDB(gff_db)
                    if self.logger is not None:
                        self.logger.info('Read gff database: {}'.format(self.db_name))
                elif os.path.exists(self.db_name) and os.path.getmtime(
                        self.db_name) < os.path.getmtime(
                            self.gff_file) and not rebuild:
                    gff_db = sqlite3.connect(self.db_name, check_same_thread=False)
                    gff_db = gffutils.FeatureDB(gff_db)
                    warnings.warn(
                        "Database file {0} is older than GFF file {1}.".format(
                            self.db_name, self.gff_file), RuntimeWarning)
                elif build_db:
                    if self.logger is not None:
                        self.logger.info('Build gff database: {}'.format(self.db_name))
                    gff_db = gffutils.create_db(self.gff_file, ':memory:', merge_strategy="create_unique", keep_order=True)
                    bk_gff = sqlite3.connect(self.db_name)
                    gff_db.conn.backup(bk_gff)
                    bk_gff.close()
                else:
                    if self.logger is not None:
                        self.logger.info('Read gff database: {}'.format(self.db_name))
                    gff_db = sqlite3.connect(self.db_name, check_same_thread=False)
                    gff_db = gffutils.FeatureDB(gff_db)
            except Exception:
                # Handle potential exceptions
                raise
        return gff_db

    def build_genome(self, rebuild=False):
        if os.path.exists(self.feature_file) and os.path.getmtime(
                self.feature_file) >= os.path.getmtime(self.genome_file):
            with bgzf.open(self.feature_file, 'r') as handle:
                self.features = [feature for feature in SeqIO.parse(handle, "fasta")]
                if self.logger is not None:
                    self.logger.info('Opened file: {}'.format(self.feature_file))
                    self.logger.info('Memory usage: {} MB'.format(memory_usage()))
        elif os.path.exists(self.feature_file) and os.path.getmtime(
                self.feature_file) < os.path.getmtime(
                    self.genome_file) and not rebuild:
            with bgzf.open(self.feature_file, 'r') as handle:
                self.features = [feature for feature in SeqIO.parse(handle, "fasta")]
                if self.logger is not None:
                    self.logger.info('Opened file: {}'.format(self.feature_file))
                    self.logger.info('Memory usage: {} MB'.format(memory_usage()))
            warnings.warn(
                "Feature file {0} is older than genome file {1}.".format(
                    self.feature_file, self.genome_file), RuntimeWarning)

    def _write_features():
        if self.features is None:
            return
        with bgzf.open(self.features_file, 'w') as handle:
            SeqIO.write(self.features, handle, "fasta")
        return

    def __iter__(self):
        for feature in features:
            yield feature

class KmerGenerator:

    def __init__(self, genome_file, gff_file, k_low, k_high, rand_seed, genome_dir=None, rebuild=False, build_db=True, logger=None):
        self.genome_file = genome_file
        self.k_low = k_low
        self.k_high = k_high
        self.rand_seed = rand_seed
        self.logger = logger 
        self.iter_count = 0
        self.gff_file = gff_file
        self.genome_dir = os.path.dirname(self.genome_file) if genome_dir is None else genome_dir
        self.genome = Genome(self.genome_dir, self.genome_file, gff_file)
        self.genome.build_genome()

    def _seq_fragmenter(self, seq):
        """
        Split a sequence into small sequences based on some criteria, e.g. 'N' characters
        """
        return filter(bool, re.split(r'[^ACGTacgt]+', str(seq).upper()))

    def _sliding_kmer(self, rng, seq):
        return [seq[i: i + rng.randint(self.k_low, self.k_high + 1)] for i in range(len(seq) - self.k_high + 1)]

    def _generator(self, rng):
        if self.logger is not None:
            self.logger.addHandler(TqdmLoggingHandler())
        for i, features in tqdm(enumerate(self.genome.features), total=len(self.genome.features)):
            yield features.seq, i

    def __len__(self):
        return len(self.genome.features)

    def __iter__(self):
        self.iter_count += 1
        rng = np.random.RandomState(self.rand_seed)
        for seq, i in self._generator(rng):
            n_seq_splits = list(self._seq_fragmenter(seq))
            # self.logger.debug('Splits of len={} to: {}'.format(len(seq), [len(f) for f in acgt_seq_splits]))
            for s_seq in n_seq_splits:
                kmer_seq = self._sliding_kmer(rng, s_seq)  # list of strings
                if len(kmer_seq) > 0:
                    yield TaggedDocument(kmer_seq, [i])
