import os
import resource
import argparse
import itertools
import re
import gzip
import numpy as np
from enum import Enum
from typing import Dict, List
from Bio import SeqIO


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

    def __init__(self, k_low, k_high, rand_seed, data_dir="", filenames=[], logger=None, seqlen_ulim=5000):
        self.filenames = filenames
        self.k_low = k_low
        self.k_high = k_high
        self.seqlen_ulim = seqlen_ulim
        self.rand_seed = rand_seed
        self.logger = logger 
        self.iter_count = 0
        self.filenames.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith((".fasta", ".fa", ".fa.gz"))])

    def _seq_fragmenter(self, seq):
        """
        Split a sequence into small sequences based on some criteria, e.g. 'N' characters
        """
        return filter(bool, re.split(r'[^ACGTacgt]+', str(seq).upper()))

    def _sliding_kmer(self, rng, seq):
        return [seq[i: i + rng.randint(self.k_low, self.k_high + 1)] for i in range(len(seq) - self.k_high + 1)]

    def _filehandle_generator(self):
        for filename in self.filenames:
            # unpacking the tuple
            file_name, file_extension = os.path.splitext(filename)
            if file_extension == '.gz':
                with gzip.open(filename, "rt") as file:
                    self.logger.info('Opened file: {}'.format(filename))
                    self.logger.info('Memory usage: {} MB'.format(memory_usage()))
                    yield file
            else:
                with open(filename, "rt") as file:
                    self.logger.info('Opened file: {}'.format(filename))
                    self.logger.info('Memory usage: {} MB'.format(memory_usage()))
                    yield file

    def _generator(self, rng):
        for fh in self._filehandle_generator():
            # SeqIO takes twice as much memory than even simple fh.readlines()
            for seq_record in SeqIO.parse(fh, "fasta"):
                whole_seq = seq_record.seq
                # self.logger.info('Whole fasta seqlen: {}'.format(len(whole_seq)))
                curr_left = 0
                while curr_left < len(whole_seq):
                    seqlen = rng.randint(self.seqlen_ulim // 2, self.seqlen_ulim)
                    segment = seq_record.seq[curr_left: seqlen + curr_left]
                    curr_left += seqlen
                    # self.logger.debug('input seq len: {}'.format(len(segment)))
                    yield segment

    def __iter__(self):
        self.iter_count += 1
        rng = np.random.RandomState(self.rand_seed)
        for seq in self._generator(rng):
            acgt_seq_splits = list(self._seq_fragmenter(seq))
            # self.logger.debug('Splits of len={} to: {}'.format(len(seq), [len(f) for f in acgt_seq_splits]))

            for acgt_seq in acgt_seq_splits:
                kmer_seq = self._sliding_kmer(rng, acgt_seq)  # list of strings
                if len(kmer_seq) > 0:
                    if self.iter_count == 1:
                        # only collect stats on the first call
                        # self.histogram.add(kmer_seq)
                        pass
                    yield kmer_seq
