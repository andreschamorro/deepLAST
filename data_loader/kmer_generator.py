import os
import resource
import argparse
import itertools
import re
import gzip
from threading import Lock
import numpy as np
from enum import Enum
from typing import Dict, List
import gffutils
from pyfaidx import Fasta
import sqlite3
from gensim.models.doc2vec import TaggedDocument


def memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E6

class BaseSet(Enum):
    """
    Enum for Base set types.
    """

    Pure: str = 'pure'
    Dubious2: str = 'dubious2'
    Dubious3: str = 'dubious3'
    Full: str = 'full'

class KmerGenerator:

    def __init__(self, fasta_file, gff_file, k_low, k_high, rand_seed, rebuild=False, build_db=True, logger=None):
        self.fasta_file = fasta_file
        self.k_low = k_low
        self.k_high = k_high
        self.rand_seed = rand_seed
        self.logger = logger 
        self.iter_count = 0
        self.lock = Lock()
        self.gff_file = gff_file
        self._fasta = Fasta(filename=fasta_file, sequence_always_upper=True)
        if self.logger is not None:
            self.logger.info('Opened file: {}'.format(fasta_file))
            self.logger.info('Memory usage: {} MB'.format(memory_usage()))
        self.gff_db = self.build_db(rebuild, build_db)

    def build_db(self, rebuild=False, build_db=True):
        self.db_name = os.path.splitext(self.gff_file)[0] + '.db'
        with self.lock:  # lock around index generation so only one thread calls method
            try:
                if os.path.exists(self.db_name) and os.path.getmtime(
                        self.db_name) >= os.path.getmtime(self.gff_file):
                    gff_db = gffutils.FeatureDB(self.db_name)
                    print("Read db")
                elif os.path.exists(self.db_name) and os.path.getmtime(
                        self.db_name) < os.path.getmtime(
                            self.gff_file) and not rebuild:
                    gff_db = gffutils.FeatureDB(self.db_name)
                    warnings.warn(
                        "Database file {0} is older than GFF file {1}.".format(
                            self.db_name, self.gff_file), RuntimeWarning)
                elif build_db:
                    print("Build db")
                    gff_db = gffutils.create_db(self.gff_file, ':memory:', merge_strategy="create_unique", keep_order=True)
                    bk_gff = sqlite3.connect(self.db_name)
                    gff_db.conn.backup(bk_gff)
                    bk_gff.close()
                else:
                    print("Default")
                    gff_db = gffutils.FeatureDB(self.db_name)
            except Exception:
                # Handle potential exceptions
                raise
        return gff_db

    def _seq_fragmenter(self, seq):
        """
        Split a sequence into small sequences based on some criteria, e.g. 'N' characters
        """
        return filter(bool, re.split(r'[^ACGTacgt]+', str(seq).upper()))

    def _sliding_kmer(self, rng, seq):
        return [seq[i: i + rng.randint(self.k_low, self.k_high + 1)] for i in range(len(seq) - self.k_high + 1)]

    def _children(self, fdb, s_type, order_by='start', reverse=False):
        for s in self.gff_db.children(fdb, featuretype=s_type, order_by=order_by, reverse=reverse):
            yield s.sequence(self._fasta, use_strand=True), s.id

    def _generator(self, f_type='transcript', s_type='exon', order_by='start'):
        for fdb in self.gff_db.features_of_type(f_type, order_by=order_by): # or mRNA depending on the gff
            for seq, seqid in self._children(fdb, s_type, order_by=order_by, reverse=(fdb.strand == '-')):
                yield seq, seqid, fdb.id

    def __iter__(self):
        self.iter_count += 1
        rng = np.random.RandomState(self.rand_seed)
        for seq, seqid, _ in self._generator():
            n_seq_splits = list(self._seq_fragmenter(seq))
            # self.logger.debug('Splits of len={} to: {}'.format(len(seq), [len(f) for f in acgt_seq_splits]))
            for s_seq in n_seq_splits:
                kmer_seq = self._sliding_kmer(rng, s_seq)  # list of strings
                if len(kmer_seq) > 0:
                    if self.iter_count == 1:
                        # only collect stats on the first call
                        # self.histogram.add(kmer_seq)
                        pass
                    yield TaggedDocument(kmer_seq, seqid)
