import logging
import os
import re
import numpy as np
from Bio import bgzf, SeqIO
from gensim.models.doc2vec import TaggedDocument
from gensim import utils

_base_to_int = {
        'T': 0, 't': 0,
        'C': 1, 'c': 1,
        'A': 2, 'a': 2,
        'G': 3, 'g': 3,
        'N': 4} # Like T in two bits
_bases = ['T', 'C', 'A', 'G']

OOV_ID = -1

def _standardization(sequence):
    return re.sub(r'[^actg]', '', sequence.lower())

def _read(bgz_file):
    with bgzf.open(bgz_file, 'r') as fa:
        for i, feature in enumerate(SeqIO.parse(fa, "fasta")):
            yield i, _standardization(str(feature.seq))

def _kmer_tokenizer(sequence, k):
    _mask = np.uint64((np.uint64(1) << np.uint64(2*k))-1)
    kmer = np.uint64(0)
    l = 0
    for n in sequence:
        kmer = (kmer << np.uint64(2) | np.uint64(_base_to_int[n])) & _mask
        l += 1
        if (l >= k):
            yield kmer

def decode(code, length):
    ret = ''
    for _ in range(length):
        index = code & np.uint64(3)
        code >>= np.uint64(2)
        ret = _bases[index] + ret
    return ret

def trns_by_id(bgz_file):
    return {i: s for i, s in _read(bgz_file)}

def read_trns(bgz_file, k, tokens_only=False):
    for t_id, seq in trns_by_id(bgz_file).items():
        if tokens_only:
            yield list(map(str, _kmer_tokenizer(seq, k)))
        else:
            yield TaggedDocument(list(_kmer_tokenizer(seq, k)), [t_id])

class TrnsIterator():
    def __init__(self, bgz_file, k, tokens_only=False):
        self.generator_function = read_trns 
        self.bgz_file = bgz_file
        self.k = k
        self.tokens_only = tokens_only
        self.generator = self.generator_function(self.bgz_file, self.k, self.tokens_only) 

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function(self.bgz_file, self.k, self.tokens_only)
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result

class TaggedKFasta:
    def __init__(self, handles, k, formats="fasta", cipher=lambda i: i):
        """Iterate over a file that contains documents:
        one line = :class:`~gensim.models.doc2vec.TaggedDocument` object.
        Words are expected to be already preprocessed and separated by whitespace. Document tags are constructed
        automatically from the document line number (each document gets a unique integer tag).
        Parameters
        ----------
        handle : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).
        Examples
        --------
        .. sourcecode:: pycon
            >>> 
            >>> 
            ...     pass
        """
        self.handles = handles
        self.k = k
        if isinstance(formats, str):
            self.formats = [formats] * len(handles)
        else:
            self.formats = formats
        self.prefetched = False
        self.data = []
        self.cipher = cipher
        
    def prefetch(self):
        for handle, fts in zip(self.handles, self.formats):
            try:
                # Assume it is a file-like object and try treating it as such
                # Things that don't have seek will trigger an exception
                handle.seek(0)
                for feature in SeqIO.parse(handle, fts):
                    self.data.append((feature.id, list(map(
                        str, _kmer_tokenizer(_standardization(str(feature.seq)), self.k)))))
            except AttributeError:
                # If it didn't work like a file, use it as a string filename
                with utils.open(handle, 'r') as fin:
                    for feature in SeqIO.parse(fin, fts):
                        self.data.append((feature.id, list(map(
                            str, _kmer_tokenizer(_standardization(str(feature.seq)), self.k)))))
        self.prefetched = True
        return self

    def shuffle(self):
        np.random.shuffle(self.data)
        return self

    def get_generator(self, tokens_only=False):
        for sid, tokens in self.data:
            yield tokens if tokens_only else TaggedDocument(tokens, [self.cipher(sid)])

    def __iter__(self):
        """Iterate through the lines in the source.
        Yields
        ------
        :class:`~gensim.models.doc2vec.TaggedDocument`
            Document from `source` specified in the constructor.
        """
        for sid, tokens in self.data:
            yield TaggedDocument(tokens, [self.cipher(sid)])

class TaggedKPairRead:
    def __init__(self, handles, k, formats="fastq", cipher=lambda i: i):
        """Iterate over a file that contains documents:
        one line = :class:`~gensim.models.doc2vec.TaggedDocument` object.
        Words are expected to be already preprocessed and separated by whitespace. Document tags are constructed
        automatically from the document line number (each document gets a unique integer tag).
        Parameters
        ----------
        handle : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).
        Examples
        --------
        .. sourcecode:: pycon
            >>> 
            >>> 
            ...     pass
        """
        self.handles = handles
        self.k = k
        if isinstance(formats, str):
            self.formats = [formats] * len(handles)
        else:
            self.formats = formats
        self.prefetched = False
        self.data = []
        self.cipher = cipher
        
    def prefetch(self):
        for (hr1, hr2), fts in zip(self.handles, self.formats):
            try:
                # Assume it is a file-like object and try treating it as such
                # Things that don't have seek will trigger an exception
                hr1.seek(0)
                hr2.seek(0)
                for f1, f2 in zip(SeqIO.parse(hr1, fts), SeqIO.parse(hr2, fts)):
                    self.data.append((f1.id, list(map(
                            str, [*_kmer_tokenizer(_standardization(str(f1.seq)), self.k), 
                                    *_kmer_tokenizer(_standardization(str(f2.seq)), self.k)]))))
            except AttributeError:
                # If it didn't work like a file, use it as a string filename
                with utils.open(hr1, 'r') as fr1, utils.open(hr2, 'r') as fr2:
                    for f1, f2 in zip(SeqIO.parse(fr1, fts), SeqIO.parse(fr2, fts)):
                        self.data.append((f1.id, list(map(
                                str, [*_kmer_tokenizer(_standardization(str(f1.seq)), self.k), 
                                        *_kmer_tokenizer(_standardization(str(f2.seq)), self.k)]))))
        self.prefetched = True
        return self

    def shuffle(self):
        np.random.shuffle(self.data)
        return self

    def sample(self, n):
        pr = TaggedKPairRead(self.handles, self.k, self.formats, self.cipher)
        pr.data = [self.data[i] for i in np.random.randint(len(self.data), size=n)]
        return pr 

    def get_generator(self, tokens_only=False):
        for sid, tokens in self.data:
            yield tokens if tokens_only else TaggedDocument(tokens, [self.cipher(sid)])

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        """Iterate through the lines in the source.
        Yields
        ------
        :class:`~gensim.models.doc2vec.TaggedDocument`
            Document from `source` specified in the constructor.
        """
        for sid, tokens in self.data:
            yield TaggedDocument(tokens, [self.cipher(sid)])
