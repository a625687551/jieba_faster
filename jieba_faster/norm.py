# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool

from . import Tokenizer
"""
这里前期重写后期，采用继承
"""


class Cutter(object):
    def __init__(self, dictionary="dict.txt", multi_process=False,
                 process_num=2):
        self.dictionary = dictionary
        self.token = Tokenizer(self.dictionary)
        self.multi_process = multi_process
        if multi_process:
            self.pool = Pool(process_num)

    def cut(self, sentence, cut_all=False, hmm=True):
        if self.multi_process:
            return self._pcut(sentence, cut_all, hmm)
        return self.token.cut(self, sentence, cut_all, hmm)

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def lcut_all(self, sentence):
        return self.lcut(sentence, True)

    def lcut_no_hmm(self, sentence):
        return self.lcut(sentence, False, False)

    def _pcut(self, sentence, cut_all=False, HMM=True):
        parts = sentence.splitlines(True)
        if cut_all:
            result = self.pool.map(self.token._lcut_all, parts)
        elif HMM:
            result = self.pool.map(self.token.lcut, parts)
        else:
            result = self.pool.map(self.lcut_no_hmm, parts)
        for r in result:
            for w in r:
                yield w

    def enable_parallel(self):
        if os.name == "nt":
            raise Exception("jieba: parallel mode only supports posix system")
        self.token.check_initialized()
