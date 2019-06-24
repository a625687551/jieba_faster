from __future__ import absolute_import, unicode_literals

__version__ = '0.39'
__license__ = 'MIT'

import re
import os
import sys
import time
import logging
import marshal
import tempfile
import threading
from math import log
from hashlib import md5
from ._compat import *
from . import finalseg

if os.name == 'nt':
    from shutil import move as _replace_file
else:
    _replace_file = os.rename

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

DICT_WRITING = {}

pool = None

re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)

re_eng = re.compile('[a-zA-Z0-9]', re.U)

# \u4E00-\u9FA5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
# re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
# Adding "-" symbol in re_han_default
# 注意正则表达式是用括号括起来的,即捕获括号
# 用括号将正则表达式括起来，那么匹配的字符串也会被列入到list中返回
# re.U是使得compile的结果取决于unicode定义的字符属性

# 精准模式正则
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)

# 全局模式的正则
re_han_cut_all = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.U)


def setLogLevel(log_level):
    global logger
    default_logger.setLevel(log_level)


class Tokenizer(object):
    """
     弃用trie树，减少内存，详情见
     https://github.com/fxsjy/jieba/pull/187
    """

    def __init__(self, dictionary=DEFAULT_DICT):
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = _get_abs_path(dictionary)
        self.FREQ = {}
        self.total = 0
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    def __repr__(self):
        return '<Tokenizer dictionary=%r>' % self.dictionary

    @staticmethod
    def gen_pfdict(filename):
        """
        构建前缀字典的核心函数
        解析离线统计词典文本文件，每一行分别对应着词、词频、词性，将词和词频提取出来，
        以词为key，以词频为value，加入到前缀词典中。对于每个词，再分别获取它的前缀词，
        如果前缀词已经存在于前缀词典中，则不处理；如果该前缀词不在前缀词典中，
        则将其词频置为0，便于后续构建有向无环图。
        Args:
            filename: file name

        Returns:

        """
        lfreq = {}  # 字典存储  词：词频
        ltotal = 0  # 所有词条出现的总次数
        f_name = resolve_filename(filename)  # 打开文件dict.txt,这里需要确认下
        for lineno, line in enumerate(filename, 1):
            try:
                line = line.strip().decode('utf-8')  # 解码为unicode
                word, freq = line.split(' ')[:2]  # 获得词条及其出现的次数
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq
                for ch in xrange(len(word)):  # 处理word的前缀
                    wfrag = word[:ch + 1]
                    if wfrag not in lfreq:  # word前缀不再lfreq则其出现频次置为0
                        lfreq[wfrag] = 0
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (
                        f_name, lineno, line))
        filename.close()
        return lfreq, ltotal

    def initialize(self, dictionary=None):
        """
        初始化，构建前缀字典
        Args:
            dictionary:

        Returns:

        """
        if dictionary:
            abs_path = _get_abs_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                return
            else:
                self.dictionary = abs_path
                self.initialized = False
        else:
            abs_path = self.dictionary

        with self.lock:
            try:
                with DICT_WRITING[abs_path]:
                    pass
            except KeyError:
                pass
            if self.initialized:
                return

            default_logger.debug("Building prefix dict from %s ..." % (
                    abs_path or 'the default dictionary'))
            t1 = time.time()
            if self.cache_file:
                cache_file = self.cache_file
            # default dictionary
            elif abs_path == DEFAULT_DICT:
                cache_file = "jieba.cache"
            # custom dictionary
            else:
                cache_file = "jieba.u%s.cache" % md5(
                    abs_path.encode('utf-8', 'replace')).hexdigest()
            cache_file = os.path.join(
                self.tmp_dir or tempfile.gettempdir(), cache_file)
            # prevent absolute path in self.cache_file
            tmpdir = os.path.dirname(cache_file)

            load_from_cache_fail = True
            if os.path.isfile(cache_file) and (abs_path == DEFAULT_DICT or
                                               os.path.getmtime(
                                                   cache_file) > os.path.getmtime(
                        abs_path)):
                default_logger.debug(
                    "Loading model from cache %s" % cache_file)
                try:
                    with open(cache_file, 'rb') as cf:
                        self.FREQ, self.total = marshal.load(cf)
                    load_from_cache_fail = False
                except Exception:
                    load_from_cache_fail = True

            if load_from_cache_fail:
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                with wlock:
                    # 构建前缀词典的入口
                    self.FREQ, self.total = self.gen_pfdict(
                        self.get_dict_file())
                    default_logger.debug(
                        "Dumping model to file cache %s" % cache_file)
                    try:
                        # prevent moving across different filesystems
                        fd, fpath = tempfile.mkstemp(dir=tmpdir)
                        with os.fdopen(fd, 'wb') as temp_cache_file:
                            marshal.dump(
                                (self.FREQ, self.total), temp_cache_file)
                        _replace_file(fpath, cache_file)
                    except Exception:
                        default_logger.exception("Dump cache file failed.")

                try:
                    del DICT_WRITING[abs_path]
                except KeyError:
                    pass

            self.initialized = True
            default_logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1))
            default_logger.debug("Prefix dict has been built successfully.")

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def calc(self, sentence, DAG, route):
        """
        动态规划，计算最大概率的切分组合
        Args:
            sentence:句子
            DAG:
            route:

        Returns:

        """
        N = len(sentence)
        route[N] = (0, 0)
        # 对概率值取对数之后的结果(可以让概率相乘的计算变成对数相加,防止相乘造成下溢)
        logtotal = log(self.total)
        # 从后往前遍历句子 反向计算最大概率
        for idx in xrange(N - 1, -1, -1):
            # 列表推倒求最大概率对数路径
            # route[idx] = max([ (概率对数，词语末字位置) for x in DAG[idx] ])
            # 以idx:(概率对数最大值，词语末字位置)键值对形式保存在route中
            # route[x+1][0] 表示 词路径[x+1,N-1]的最大概率对数,
            # [x+1][0]即表示取句子x+1位置对应元组(概率对数，词语末字位置)的概率对数
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    def get_DAG(self, sentence):
        """
        DAG中是以{key:list,...}的字典结构存储，key是字的开始位置
        有向无环图构建的核心函数
        Args:
            sentence:句子

        Returns:

        """
        self.check_initialized()
        DAG = {}  # 存储数据，格式dict
        N = len(sentence)
        for k in xrange(N):
            tmplist = []
            i = k
            frag = sentence[k]  # 位置k形成的片段
            # 判断片段是否在前缀词典中
            # 如果片段不在前缀词典中，则跳出本循环
            # 也即该片段已经超出统计词典中该词的长度
            while i < N and frag in self.FREQ:
                # 判断片段是否在前缀词典中
                # 如果片段不在前缀词典中，则跳出本循环
                # 也即该片段已经超出统计词典中该词的长度
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1  # 片段位置末尾+1
                frag = sentence[k:i + 1]  # 新的片段较旧的片段右边新增一个字
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def __cut_all(self, sentence):
        dag = self.get_DAG(sentence)
        old_j = -1
        for k, L in iteritems(dag):
            if len(L) == 1 and k > old_j:
                yield sentence[k:L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j

    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''

    def __cut_DAG(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            # buf收集连续的单个字,把它们组合成字符串再交由 finalseg.cut函数来进行下一步分词
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    else:
                        if not self.FREQ.get(buf):  # 未登录词，利用HMM
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):  # 未登录词，利用HMM
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(self, sentence, cut_all=False, HMM=True):
        """
        The main function that segments an entire sentence that contains
        Chinese characters into separated words.
        Args:
            sentence: The str(unicode) to be segmented.
            cut_all: Model type. True for full pattern, False for accurate pattern.
            HMM: Whether to use the Hidden Markov Model.

        Returns:

        """
        sentence = strdecode(sentence)
        # 不同模式下的正则
        if cut_all:
            re_han = re_han_cut_all
            re_skip = re_skip_cut_all
        else:
            re_han = re_han_default
            re_skip = re_skip_default
        # 设置不同模式下的cut_block分词方法
        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        # 先用正则对句子进行切分
        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            if re_han.match(blk):  # re_han匹配的串
                for word in cut_block(blk):  # 根据不同模式的方法来分词
                    yield word
            else:  # 按照re_skip正则表对blk进行重新切分
                tmp = re_skip.split(blk)  # 返回list
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:  # 精准模式下逐个字符输出
                        for xx in x:
                            yield xx
                    else:
                        yield x

    def cut_for_search(self, sentence, HMM=True):
        """
        分词的(HMM or no HMM)的基础上，对长词再次切分
        Finer segmentation for search engines.
        """
        words = self.cut(sentence, HMM=HMM)
        for w in words:
            if len(w) > 2:
                for i in xrange(len(w) - 1):
                    gram2 = w[i:i + 2]
                    if self.FREQ.get(gram2):
                        yield gram2
            if len(w) > 3:
                for i in xrange(len(w) - 2):
                    gram3 = w[i:i + 3]
                    if self.FREQ.get(gram3):
                        yield gram3
            yield w

    def lcut(self, *args, **kwargs):
        """cut函数，返回list版本"""
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs):
        """cut_for_search函数的返回list版本"""
        return list(self.cut_for_search(*args, **kwargs))

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence):
        """return list version"""
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence):
        """return list version"""
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence):
        """return list version"""
        return self.lcut_for_search(sentence, False)

    def get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            return get_module_res(DEFAULT_DICT_NAME)
        else:
            return open(self.dictionary, 'rb')

    def load_userdict(self, f):
        """
        Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their ocurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        """
        self.check_initialized()
        if isinstance(f, string_types):
            f_name = f
            f = open(f, 'rb')
        else:
            f_name = resolve_filename(f)
        for lineno, ln in enumerate(f, 1):
            line = ln.strip()
            if not isinstance(line, text_type):
                try:
                    line = line.decode('utf-8').lstrip('\ufeff')
                except UnicodeDecodeError:
                    raise ValueError(
                        'dictionary file %s must be utf-8' % f_name)
            if not line:
                continue
            # match won't be None because there's at least one character
            word, freq, tag = re_userdict.match(line).groups()
            if freq is not None:
                freq = freq.strip()
            if tag is not None:
                tag = tag.strip()
            self.add_word(word, freq, tag)

    def add_word(self, word, freq=None, tag=None):
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        self.FREQ[word] = freq
        self.total += freq
        if tag:
            self.user_word_tag_tab[word] = tag
        for ch in xrange(len(word)):
            wfrag = word[:ch + 1]
            if wfrag not in self.FREQ:
                self.FREQ[wfrag] = 0
        if freq == 0:
            finalseg.add_force_split(word)

    def del_word(self, word):
        """
        Convenient function for deleting a word.
        """
        self.add_word(word, 0)

    def suggest_freq(self, segment, tune=False):
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        """
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1
        if isinstance(segment, string_types):
            word = segment
            for seg in self.cut(word, HMM=False):
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        else:
            segment = tuple(map(strdecode, segment))
            word = ''.join(segment)
            for seg in segment:
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = min(int(freq * self.total), self.FREQ.get(word, 0))
        if tune:
            add_word(word, freq)
        return freq

    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        Parameter:
            - sentence: the str(unicode) to be segmented.
            - mode: "default" or "search", "search" is for finer segmentation.
            - HMM: whether to use the Hidden Markov Model.
        """
        if not isinstance(unicode_sentence, text_type):
            raise ValueError("jieba: the input parameter should be unicode.")
        start = 0
        if mode == 'default':
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                if len(w) > 2:
                    for i in xrange(len(w) - 1):
                        gram2 = w[i:i + 2]
                        if self.FREQ.get(gram2):
                            yield (gram2, start + i, start + i + 2)
                if len(w) > 3:
                    for i in xrange(len(w) - 2):
                        gram3 = w[i:i + 3]
                        if self.FREQ.get(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dictionary_path):
        with self.lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise Exception("jieba: file does not exist: " + abs_path)
            self.dictionary = abs_path
            self.initialized = False


# default Tokenizer instance

dt = Tokenizer()

# global functions

get_FREQ = lambda k, d=None: dt.FREQ.get(k, d)
add_word = dt.add_word
calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut(s):
    return dt._lcut(s)


def _lcut_no_hmm(s):
    return dt._lcut_no_hmm(s)


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut_for_search(s):
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s):
    return dt._lcut_for_search_no_hmm(s)


def _pcut(sentence, cut_all=False, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def _pcut_for_search(sentence, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def enable_parallel(processnum=None):
    """
    Change the module's `cut` and `cut_for_search` functions to the
    parallel version.

    Note that this only works using dt, custom Tokenizer
    instances are not supported.
    """
    global pool, dt, cut, cut_for_search
    from multiprocessing import cpu_count
    if os.name == 'nt':
        raise NotImplementedError(
            "jieba: parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel():
    global pool, dt, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
