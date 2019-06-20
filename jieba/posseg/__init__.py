from __future__ import absolute_import, unicode_literals
import os
import re
import sys
import jieba
import pickle
from .._compat import *
from .viterbi import viterbi

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"

re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile("(\r\n|\s)")

re_eng = re.compile("[a-zA-Z0-9]+")
re_num = re.compile("[\.0-9]+")

re_eng1 = re.compile('^[a-zA-Z0-9]$', re.U)


def load_model():
    # For Jython
    start_p = pickle.load(get_module_res("posseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("posseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("posseg", PROB_EMIT_P))
    state = pickle.load(get_module_res("posseg", CHAR_STATE_TAB_P))
    return state, start_p, trans_p, emit_p


if sys.platform.startswith("java"):
    char_state_tab_P, start_P, trans_P, emit_P = load_model()
else:
    from .char_state_tab import P as char_state_tab_P
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P


class pair(object):
    """定义一个词和词性的组合类型"""

    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __unicode__(self):
        return '%s/%s' % (self.word, self.flag)

    def __repr__(self):
        return 'pair(%r, %r)' % (self.word, self.flag)

    def __str__(self):
        if PY2:
            return self.__unicode__().encode(default_encoding)
        else:
            return self.__unicode__()

    def __iter__(self):
        return iter((self.word, self.flag))

    def __lt__(self, other):
        return self.word < other.word

    def __eq__(self, other):
        return isinstance(other,
                          pair) and self.word == other.word and self.flag == other.flag

    def __hash__(self):
        return hash(self.word)

    def encode(self, arg):
        return self.__unicode__().encode(arg)


class POSTokenizer(object):
    """
    词性分词
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or jieba.Tokenizer()
        self.load_word_tag(self.tokenizer.get_dict_file())

    def __repr__(self):
        return '<POSTokenizer tokenizer=%r>' % self.tokenizer

    def __getattr__(self, name):
        if name in ('cut_for_search', 'lcut_for_search', 'tokenize'):
            # may be possible?
            raise NotImplementedError
        return getattr(self.tokenizer, name)

    def initialize(self, dictionary=None):
        self.tokenizer.initialize(dictionary)
        self.load_word_tag(self.tokenizer.get_dict_file())

    def load_word_tag(self, f):
        self.word_tag_tab = {}
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError(
                    'invalid POS dictionary entry in %s at Line %s: %s' % (
                        f_name, lineno, line))
        f.close()

    def makesure_userdict_loaded(self):
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence):
        """
        利用HMM进行词性标注的执行函数
        Args:
            sentence:

        Returns:

        """
        # 执行viterbi算法
        prob, pos_list = viterbi(
            sentence, char_state_tab_P, start_P, trans_P, emit_P)
        begin, nexti = 0, 0

        for i, char in enumerate(sentence):
            pos = pos_list[i][0]
            # 根据状态分词
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield pair(sentence[begin:i + 1], pos_list[i][1])
                nexti = i + 1
            elif pos == 'S':
                yield pair(char, pos_list[i][1])
                nexti = i + 1
        if nexti < len(sentence):
            yield pair(sentence[nexti:], pos_list[nexti][1])

    def __cut_detail(self, sentence):
        """
        利用HMM进行词性标注的主函数
        首先利用正则表达式对未登录词组成的句子进行分割，然后根据正则表达式进行判断，
        如果匹配上，则利用隐马尔科夫模型对其进行词性标注；否则，进一步根据正则表达式，判断其类型。
        Args:
            sentence:

        Returns:

        """
        # 根据正则表达式对未登录词组成的句子进行分割
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            if re_han_detail.match(blk):  # 匹配上正则表达式
                for word in self.__cut(blk):  # 利用HMM对其标注
                    yield word
            else:  # 没有匹配上正则表达式
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        if re_num.match(x):  # 匹配为数字
                            yield pair(x, 'm')
                        elif re_eng.match(x):  # 匹配为英文
                            yield pair(x, 'eng')
                        else:  # 未知类型
                            yield pair(x, 'x')

    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng1.match(l_word):
                buf += l_word
                x = y
            else:
                if buf:
                    yield pair(buf, 'eng')
                    buf = ''
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
                x = y
        if buf:
            yield pair(buf, 'eng')
            buf = ''

    def __cut_DAG(self, sentence):
        """
        构建有向无环图
        Args:
            sentence:

        Returns:

        """
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        # 计算最大概率路径
        self.tokenizer.calc(sentence, DAG, route)

        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        # 词--词性词典中有该词，则将词性赋予给该词；否则为“x”
                        yield pair(buf, self.word_tag_tab.get(buf, 'x'))
                    # 前缀词典中没有该词，则利用HMM来进行词性标注
                    elif not self.tokenizer.FREQ.get(buf):
                        recognized = self.__cut_detail(buf)
                        for t in recognized:
                            yield t
                    else:  # 两种都不满足，则将词性标注为 x
                        for elem in buf:
                            yield pair(elem, self.word_tag_tab.get(elem, 'x'))
                    buf = ''
                # 默认将词性标注为x
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
            x = y

        if buf:
            if len(buf) == 1:
                yield pair(buf, self.word_tag_tab.get(buf, 'x'))
            elif not self.tokenizer.FREQ.get(buf):
                recognized = self.__cut_detail(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield pair(elem, self.word_tag_tab.get(elem, 'x'))

    def __cut_internal(self, sentence, HMM=True):
        self.makesure_userdict_loaded()
        sentence = strdecode(sentence)
        blocks = re_han_internal.split(sentence)
        # 是否采用HMM
        if HMM:
            cut_blk = self.__cut_DAG
        else:
            cut_blk = self.__cut_DAG_NO_HMM

        for blk in blocks:
            # 匹配汉字的正则表达式，进一步根据分割函数进行分割
            if re_han_internal.match(blk):
                for word in cut_blk(blk):
                    yield word
            # 没有匹配上汉字的正则表达式
            else:
                tmp = re_skip_internal.split(blk)
                for x in tmp:
                    if re_skip_internal.match(x):
                        yield pair(x, 'x')
                    else:
                        for xx in x:
                            if re_num.match(xx):  # 匹配为数字
                                yield pair(xx, 'm')
                            elif re_eng.match(x):  # 匹配为英文
                                yield pair(xx, 'eng')
                            else:
                                yield pair(xx, 'x')  # 匹配为未知

    def _lcut_internal(self, sentence):
        return list(self.__cut_internal(sentence))

    def _lcut_internal_no_hmm(self, sentence):
        return list(self.__cut_internal(sentence, False))

    def cut(self, sentence, HMM=True):
        for w in self.__cut_internal(sentence, HMM=HMM):
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))


# default Tokenizer instance

dt = POSTokenizer(jieba.dt)

# global functions

initialize = dt.initialize


def _lcut_internal(s):
    return dt._lcut_internal(s)


def _lcut_internal_no_hmm(s):
    return dt._lcut_internal_no_hmm(s)


def cut(sentence, HMM=True):
    """
    Global `cut` function that supports parallel processing.

    Note that this only works using dt, custom POSTokenizer
    instances are not supported.
    """
    global dt
    if jieba.pool is None:
        for w in dt.cut(sentence, HMM=HMM):
            yield w
    else:
        parts = strdecode(sentence).splitlines(True)
        if HMM:
            result = jieba.pool.map(_lcut_internal, parts)
        else:
            result = jieba.pool.map(_lcut_internal_no_hmm, parts)
        for r in result:
            for w in r:
                yield w


def lcut(sentence, HMM=True):
    return list(cut(sentence, HMM))
