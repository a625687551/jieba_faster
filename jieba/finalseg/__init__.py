from __future__ import absolute_import, unicode_literals
import re
import os
import sys
import pickle
from .._compat import *

MIN_FLOAT = -3.14e100

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"

'''
StatusSet: 
状态值(隐状态)集合有4种，分别是B,M,E,S，对应于一个汉字在词语中的地位即B（开头）,
M（中间 ),E（结尾）,S（独立成词）

ObservedSet:
观察值集合,即汉字
'''
# 状态转移集合，比如B状态前只可能是E或S状态
PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

Force_Split_Words = set([])


# 加载HMM 模型,模型参数λ=(A,B,π) 分别加载初始状态分布，转移概率, 发射概率
def load_model():
    # 初始状态概率分布
    start_p = pickle.load(get_module_res("finalseg", PROB_START_P))
    # 状态转移概率
    trans_p = pickle.load(get_module_res("finalseg", PROB_TRANS_P))
    # 观测概率分布（发射概率）
    emit_p = pickle.load(get_module_res("finalseg", PROB_EMIT_P))
    return start_p, trans_p, emit_p


if sys.platform.startswith("java"):
    start_P, trans_P, emit_P = load_model()
else:
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P
"""
HMM在实际应用中主要用来解决3类问题:
1. 评估问题(概率计算问题)
   即给定观测序列 O=O1,O2,O3…Ot和模型参数λ=(A,B,π)，怎样有效计算这一观测序列出现的概率.
   (Forward-backward算法)

2. 解码问题(预测问题)
   即给定观测序列 O=O1,O2,O3…Ot和模型参数λ=(A,B,π)，怎样寻找满足这种观察序列意义上最优的隐含状态序列S。
   (viterbi算法,近似算法)

3. 学习问题
   即HMM的模型参数λ=(A,B,π)未知，如何求出这3个参数以使观测序列O=O1,O2,O3…Ot的概率尽可能的大.
   (即用极大似然估计的方法估计参数,Baum-Welch,EM算法)
"""


# HMM模型中文分词中，我们的输入是一个句子(也就是观察值序列)，输出是这个句子中每个字的状态值

# HMM的解码问题

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # 状态概率矩阵
    path = {}
    for y in states:  # 初始化状态概率
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]  # 路径记录
    for t in xrange(1, len(obs)):  # 时刻t = 1,...,len(obs) - 1
        V.append({})
        newpath = {}
        for y in states:  # 当前时刻所处的各种可能的状态
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            # t 时刻状态为y的最大概率(从t-1时刻中选择到达时刻t且状态为y的状态y0)
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for
                 y0 in PrevStatus[y]])
            V[t][y] = prob
            newpath[y] = path[state] + [y]  # 上一刻最大状态+这一刻状态
        path = newpath
    # 求出最后一个字那种状态的对应概率最大，最后一个字只可能有两种情况：E(结尾)和S(独立词)
    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')

    return (prob, path[state])  # 返回最大概率对数和最优路径


# 利用viterbi算法得到句子分词的生成器
def __cut(sentence):
    global emit_P
    # viterbi算法得到sentence的切分
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    # 基于隐藏状态进行分词
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':  # 字所处的位置是开始位置
            begin = i
        elif pos == 'E':  # 字所处的位置是结束位置
            yield sentence[begin:i + 1]  # 这个序列就是一个分词
            nexti = i + 1
        elif pos == 'S':  # 单独成字
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]  # 剩余的直接作为一个分词返回


re_han = re.compile("([\u4E00-\u9FD5]+)")  # 匹配中文的正则
re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")  # 匹配数字（包含小数）或字母数字


def add_force_split(word):
    """
    强制切分词语，具体不太了解
    :param word:
    :return:
    """
    global Force_Split_Words
    Force_Split_Words.add(word)


def cut(sentence):
    sentence = strdecode(sentence)
    blocks = re_han.split(sentence)
    for blk in blocks:
        if re_han.match(blk):  # 汉语块
            for word in __cut(blk):  # 调用HMM切分
                if word not in Force_Split_Words:
                    yield word
                else:
                    for c in word:
                        yield c
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x
