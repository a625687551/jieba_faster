#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
import sys
from operator import itemgetter
from collections import defaultdict
import jieba.posseg
from .tfidf import KeywordExtractor
from .._compat import *


class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for x in xrange(10):  # 10 iters 默认遍历10次
            for n in sorted_keys:  # 遍历各个节点
                s = 0
                for e in self.graph[n]:  # 遍历各个节点的入度节点
                    # 将如度节点的贡献后的权值相加
                    # 贡献率 = 入度结点与结点n的共现次数 / 入度结点的所有出度的次数
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                # 更新节点n的权重
                ws[n] = (1 - self.d) + self.d * s

        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])
        # 获取权重最大和最小值
        for w in itervalues(ws):
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w
        # 归一化
        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws


class TextRank(KeywordExtractor):

    def __init__(self):
        self.tokenizer = self.postokenizer = jieba.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.pos_filt = frozenset(('ns', 'n', 'vn', 'v'))
        self.span = 5

    def pairfilter(self, wp):
        return (wp.flag in self.pos_filt and len(wp.word.strip()) >= 2
                and wp.word.lower() not in self.stop_words)

    def textrank(self, sentence, topK=20, withWeight=False,
                 allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        """
        Extract keywords from sentence using TextRank algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                        if the POS of w is not in this list, it will be filtered.
            - withFlag: if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        self.pos_filt = frozenset(allowPOS)
        g = UndirectWeightedGraph()  # 定义有向无权图
        cm = defaultdict(int)  # 定义共现词典
        words = tuple(self.tokenizer.cut(sentence))  # 分词
        for i, wp in enumerate(words):  # 遍历每个词
            if self.pairfilter(wp):  # 词i满足过滤条件
                for j in xrange(i + 1, i + self.span):  # 依次遍历词i之后窗口范围内的词
                    if j >= len(words):  # 不能超出句子长度
                        break
                    if not self.pairfilter(words[j]):  # 不能满足过滤条件，pass
                        continue
                    # 词i和词j作为key,出现的次数作为value，添加到共现词典中
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1
        # 依次遍历共现词典的每个元素，将词i，词j作为一条边起始点和终止点，共现的次数作为边的权重
        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()  # 运行text-rank算法
        # 排序
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        if topK:
            return tags[:topK]
        else:
            return tags

    extract_tags = textrank
