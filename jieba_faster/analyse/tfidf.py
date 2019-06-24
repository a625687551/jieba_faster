# encoding=utf-8
from __future__ import absolute_import
import os
import jieba
import jieba.posseg
from operator import itemgetter

_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),
                                                              os.path.dirname(
                                                                  __file__),
                                                              path))
_get_abs_path = jieba._get_abs_path

DEFAULT_IDF = _get_module_path("idf.txt")


class KeywordExtractor(object):
    STOP_WORDS = set((
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
        "by", "be", "as", "on", "with", "can", "if", "from", "which", "you",
        "it",
        "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
    ))

    def set_stop_words(self, stop_words_path):
        abs_path = _get_abs_path(stop_words_path)
        if not os.path.isfile(abs_path):
            raise Exception("jieba: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)

    def extract_tags(self, *args, **kwargs):
        raise NotImplementedError


class IDFLoader(object):

    def __init__(self, idf_path=None):
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            self.idf_freq = {}
            for line in content.splitlines():
                word, freq = line.strip().split(' ')
                self.idf_freq[word] = float(freq)
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self):
        return self.idf_freq, self.median_idf


class TFIDF(KeywordExtractor):

    def __init__(self, idf_path=None):
        """
        初始化
        Args:
            idf_path: idf路径
        """
        self.tokenizer = jieba.dt
        self.postokenizer = jieba.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path):
        new_abs_path = _get_abs_path(idf_path)
        if not os.path.isfile(new_abs_path):
            raise Exception("jieba: file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(),
                     withFlag=False):
        """
        Extract keywords from sentence using TF-IDF algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr'].
                        if the POS of w is not in this list,it will be filtered.
            - withFlag: only work with allowPOS is not empty.
                        if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        if allowPOS:  # 调用词性标注接口
            allowPOS = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence)
        else:  # 普通分词
            words = self.tokenizer.cut(sentence)
        freq = {}
        for w in words:
            if allowPOS:
                if w.flag not in allowPOS:
                    continue
                elif not withFlag:
                    w = w.word
            wc = w.word if allowPOS and withFlag else w
            # 判断长度是否小于2和是否是停用词
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            freq[w] = freq.get(w, 0.0) + 1.0  # 添加到词频字典中，次数加1
            total = sum(freq.values())  # 统计词频字典的总次数
            for k in freq:
                kw = k.word if allowPOS and withFlag else k
                # 计算每一个词的tf-idf值
                freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

            if withWeight:  # 根据tf-idf进行排序
                tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
            else:
                tags = sorted(freq, key=freq.__getitem__, reverse=True)
            if topK:
                return tags[:topK]
            else:
                return tags
