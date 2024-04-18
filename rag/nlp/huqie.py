# -*- coding: utf-8 -*-

import copy
import datrie
import math
import os
import re
import string
import sys
from hanziconv import HanziConv
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from api.utils.file_utils import get_project_base_directory


class Huqie:
    def key_(self, line):
        return str(line.lower().encode("utf-8"))[2:-1]

    def rkey_(self, line):
        return str(("DD" + (line[::-1].lower())).encode("utf-8"))[2:-1]

    def loadDict_(self, fnm):
        print("[HUQIE]:Build trie", fnm, file=sys.stderr)
        try:
            of = open(fnm, "r", encoding="UTF-8")
            while True:
                line = of.readline()
                if not line:
                    break
                line = re.sub(r"[\r\n]+", "", line)
                line = re.split(r"[ \t]", line)
                k = self.key_(line[0])
                F = int(math.log(float(line[1]) / self.DENOMINATOR) + .5)
                if k not in self.trie_ or self.trie_[k][0] < F:
                    self.trie_[self.key_(line[0])] = (F, line[2])
                self.trie_[self.rkey_(line[0])] = 1
            self.trie_.save(fnm + ".trie")
            of.close()
        except Exception as e:
            print("[HUQIE]:Faild to build trie, ", fnm, e, file=sys.stderr)

    def __init__(self, debug=False):
        self.DEBUG = debug
        self.DENOMINATOR = 1000000
        self.trie_ = datrie.Trie(string.printable)
        self.DIR_ = os.path.join(get_project_base_directory(), "rag/res", "huqie")

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.SPLIT_CHAR = r"([ ,\.<>/?;'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；‘’：“”【】~！￥%……（）——-]+|[a-z\.-]+|[0-9,\.-]+)"
        try:
            self.trie_ = datrie.Trie.load(self.DIR_ + ".txt.trie")
            return
        except Exception as e:
            print("[HUQIE]:Build default trie", file=sys.stderr)
            self.trie_ = datrie.Trie(string.printable)

        self.loadDict_(self.DIR_ + ".txt")

    def loadUserDict(self, fnm):
        try:
            self.trie_ = datrie.Trie.load(fnm + ".trie")
            return
        except Exception as e:
            self.trie_ = datrie.Trie(string.printable)
        self.loadDict_(fnm)

    def addUserDict(self, fnm):
        self.loadDict_(fnm)

    def _strQ2B(self, ustring):
        """把字符串全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        return HanziConv.toSimplified(line)

    def dfs_(self, chars, s, preTks, tkslist):
        # code read:
        # 采用dfs方法获取chars字符串所有可能分词结果及分数
        # 递归逻辑应该是：
        # chars[:s]分词结果在preTks中，然后递归得到chars[s:]的所有分词结果，再与preTks
        # 进行组合得到最后结果。递归停止条件为s==len(chars)
        # 但下面的实现看起似乎有些问题
        # 
        ## 参数说明：
        ## chars：待分词的字符串
        ## s：起始处理字符的序号
        ## preTks: s之前的分词结果
        ## tkslist: 最终结果的容器，其中每一个元素是一个可能分词结果及分数
        MAX_L = 10
        res = s
        # if s > MAX_L or s>= len(chars):
        if s >= len(chars):
            tkslist.append(preTks)
            return res

        # pruning
        S = s + 1
        # if s + 2 <= len(chars):
        #     t1, t2 = "".join(chars[s:s + 1]), "".join(chars[s:s + 2])
        #     if self.trie_.has_keys_with_prefix(self.key_(t1)) and not self.trie_.has_keys_with_prefix(
        #             self.key_(t2)):
        #         # code read:
        #         # 这里的逻辑看起来是多余的，因为如果走到这里，那么在下面的for循环里面：
        #         # if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
        #         #     break
        #         # 一定第一次就会break掉，从而继续最下面的逻辑：将当前s字符单独作为一个词加入preTks，然后继续递归s+1
        #         # 这个没错，但是似乎没必要，可能是为了prune（S=s+2，从而在for循环中去掉chars[s]作为单个字符词的分支
        #         # 但是其实并没有起到prune的效果，因为虽然for循环中去掉了，但是在最下面的逻辑中还是会还是会递归递归s+1
        #         # ）
        #         # （update：经过实验：去掉这段，分词结果是一样的，且还更快，因为虽然没有在for循环中去掉chars[s]作为
        #         # 单个字符词的分支，但是从for循环中出来之后，res一定是等于len(chars)的，即>s的，所以就不会走到最下面的
        #         # 递归s+1那里的逻辑了）
        #         S = s + 2
        if len(preTks) > 2 and len(
                preTks[-1][0]) == 1 and len(preTks[-2][0]) == 1 and len(preTks[-3][0]) == 1:
            # code read:
            # 下面这段翻译过来就是如果前面三个token都是单字符的 且 当前字符chars[s]可以与前面的token连接起来有意义
            # （即在字典中存在前缀），那么在for循环中就不再考虑将当前chars[s]作为单个字符词分出来了的情况了（S=s+2）。
            # 这个确实起到了prune的作用（因为剪掉了chars[s]作为单个字符词的分支），不过感觉这种prune逻辑挺生硬的
            t1 = preTks[-1][0] + "".join(chars[s:s + 1])
            if self.trie_.has_keys_with_prefix(self.key_(t1)):
                S = s + 2

        ################
        for e in range(S, len(chars) + 1):
            # code read:
            # 这循环里面做的就是从chars[s]开始往后看，如果与后面字符链接起来有意义，那么就连起来作为token放入preTks，
            # 然后递归剩下的
            t = "".join(chars[s:e])
            k = self.key_(t)

            if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
                break

            if k in self.trie_:
                pretks = copy.deepcopy(preTks)
                if k in self.trie_:
                    # code read:
                    # 必然走到这里
                    pretks.append((t, self.trie_[k]))
                else:
                    # code read:
                    # 这段是废代码，前面已经判断过一次"if k in self.trie_"了
                    pretks.append((t, (-12, '')))
                # code read:
                # 如果走到这里，那么res一定就等于len(chars)了（因为max中的子问题的递归调用返回肯定是len(chars)）
                res = max(res, self.dfs_(chars, e, pretks, tkslist))

        if res > s:
            # code read:
            # 这里其实就是，但凡以chars[s]开头的token有意义，后面chars[s+1]的情况都不用考虑了
            # 在上面for循环里面已经都搞过了，这里实际上也是prune
            return res

        t = "".join(chars[s:s + 1])
        k = self.key_(t)
        if k in self.trie_:
            preTks.append((t, self.trie_[k]))
        else:
            preTks.append((t, (-12, '')))

        return self.dfs_(chars, s + 1, preTks, tkslist)

    def freq(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return 0
        return int(math.exp(self.trie_[k][0]) * self.DENOMINATOR + 0.5)

    def tag(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return ""
        return self.trie_[k][1]

    def score_(self, tfts):
        B = 30
        F, L, tks = 0, 0, []
        for tk, (freq, tag) in tfts:
            F += freq
            L += 0 if len(tk) < 2 else 1
            tks.append(tk)
        F /= len(tks)
        L /= len(tks)
        if self.DEBUG:
            print("[SC]", tks, len(tks), L, F, B / len(tks) + L + F)
        return tks, B / len(tks) + L + F

    def sortTks_(self, tkslist):
        res = []
        for tfts in tkslist:
            tks, s = self.score_(tfts)
            res.append((tks, s))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def merge_(self, tks):
        patts = [
            (r"[ ]+", " "),
            (r"([0-9\+\.,%\*=-]) ([0-9\+\.,%\*=-])", r"\1\2"),
        ]
        # for p,s in patts: tks = re.sub(p, s, tks)

        # if split chars is part of token
        res = []
        tks = re.sub(r"[ ]+", " ", tks).split(" ")
        s = 0
        while True:
            if s >= len(tks):
                break
            E = s + 1
            for e in range(s + 2, min(len(tks) + 2, s + 6)):
                tk = "".join(tks[s:e])
                if re.search(self.SPLIT_CHAR, tk) and self.freq(tk):
                    E = e
            res.append("".join(tks[s:E]))
            s = E

        return " ".join(res)

    def maxForward_(self, line):
        # code read:
        # 前向最大匹配分词，即从左到右，逐个字符扫描，找到最长的在词库中的词作为一个词
        # 例如ABCDE，如果词库中有AB，也有ABC，那第一个分出来的词是ABC
        # 词库是采用的trie树结构
        res = []
        s = 0
        while s < len(line):
            e = s + 1
            t = line[s:e]
            while e < len(line) and self.trie_.has_keys_with_prefix(
                    self.key_(t)):
                e += 1
                t = line[s:e]

            while e - 1 > s and self.key_(t) not in self.trie_:
                e -= 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s = e

        return self.score_(res)

    def maxBackward_(self, line):
        # code read:
        # 逆向最大匹配分词，即从左到右，逐个字符扫描，找到最长的在词库中的词作为一个词
        # 例如ABCDE，如果词库中有DE，也有CDE，那第一个分出来的词是CDE
        # 词库是采用的trie树结构
        res = []
        s = len(line) - 1
        while s >= 0:
            e = s + 1
            t = line[s:e]
            while s > 0 and self.trie_.has_keys_with_prefix(self.rkey_(t)):
                s -= 1
                t = line[s:e]

            while s + 1 < e and self.key_(t) not in self.trie_:
                s += 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s -= 1

        return self.score_(res[::-1])

    def qie(self, line):
        # code read:
        # 这个函数应该就是对line做分词，返回是把分出的词用空格连接起来的字符串
        
        line = self._strQ2B(line).lower()
        line = self._tradi2simp(line)
        zh_num = len([1 for c in line if is_chinese(c)])
        if zh_num < len(line) * 0.2:
            # code read:
            # 这个处理不知道是依据什么，如果一句话中中文字数占比小于20%就按英文分词？
            return " ".join([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(line)])

        arr = re.split(self.SPLIT_CHAR, line)
        res = []
        for L in arr:
            if len(L) < 2 or re.match(
                    r"[a-z\.-]+$", L) or re.match(r"[0-9\.-]+$", L):
                # code read:
                # 对于非中文的直接或者字符数小于2的不需要再分
                res.append(L)
                continue
            # print(L)

            # use maxforward for the first time
            # code read:
            # 先前向最大匹配分词
            tks, s = self.maxForward_(L)
            # code read:
            # 再逆向最大匹配分词
            tks1, s1 = self.maxBackward_(L)
            if self.DEBUG:
                print("[FW]", tks, s)
                print("[BW]", tks1, s1)

            # code read:
            # diff[i]表示第i个词前向和逆向分词结果是否一致，0表示一致，1表示不一致
            diff = [0 for _ in range(max(len(tks1), len(tks)))]
            for i in range(min(len(tks1), len(tks))):
                if tks[i] != tks1[i]:
                    diff[i] = 1

            # code read:
            # 采用分词结果分数更高的作为初步分词结果
            if s1 > s:
                tks = tks1

            # code read:
            # 下面进行分词二次处理：
            # 对于正向和逆向分词一致的结果保留
            # 不一致的部分使用dfs得到所有分词可能组合并取分数最高的作为进一步的分词结果
            i = 0
            while i < len(tks):
                s = i
                while s < len(tks) and diff[s] == 0:
                    # code read:
                    # 找到不一致的起始偏移序号
                    s += 1
                if s == len(tks):
                    res.append(" ".join(tks[i:]))
                    break
                if s > i:
                    # code read:
                    # 对于正向和逆向分词一致的结果保留
                    res.append(" ".join(tks[i:s]))

                # code read:
                # 截取不一致的部分：但是一次最多处理5个不一致的连续词，依据？为了控制下面dfs的复杂度？
                e = s
                while e < len(tks) and e - s < 5 and diff[e] == 1:
                    e += 1

                tkslist = []
                # code read:
                # 对于不一致的部分，采用dfs得到所有可能分词结果及每个结果分数
                # 但是这里为什么要带上后面一个一致的词（e+1），是bug还是另有原因？
                self.dfs_("".join(tks[s:e + 1]), 0, [], tkslist)
                # code read:
                # 对所有可能分词结果进行排序，取得分最高的一个
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

                i = e + 1

        res = " ".join(res)
        if self.DEBUG:
            print("[TKS]", self.merge_(res))
        return self.merge_(res)

    def qieqie(self, tks):
        # code read:
        # 这个函数就是把qie的结果再进一步用dfs切细
        tks = tks.split(" ")
        zh_num = len([1 for c in tks if c and is_chinese(c[0])])
        if zh_num < len(tks) * 0.2:
            res = []
            for tk in tks:
                res.extend(tk.split("/"))
            return " ".join(res)

        res = []
        for tk in tks:
            if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
                res.append(tk)
                continue
            tkslist = []
            if len(tk) > 10:
                tkslist.append(tk)
            else:
                self.dfs_(tk, 0, [], tkslist)
            if len(tkslist) < 2:
                res.append(tk)
                continue
            stk = self.sortTks_(tkslist)[1][0]
            if len(stk) == len(tk):
                stk = tk
            else:
                if re.match(r"[a-z\.-]+$", tk):
                    for t in stk:
                        if len(t) < 3:
                            stk = tk
                            break
                    else:
                        stk = " ".join(stk)
                else:
                    stk = " ".join(stk)

            res.append(stk)

        return " ".join(res)


def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


def is_number(s):
    if s >= u'\u0030' and s <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(s):
    if (s >= u'\u0041' and s <= u'\u005a') or (
            s >= u'\u0061' and s <= u'\u007a'):
        return True
    else:
        return False


def naiveQie(txt):
    tks = []
    for t in txt.split(" "):
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]
                            ) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


hq = Huqie()
qie = hq.qie
qieqie = hq.qieqie
tag = hq.tag
freq = hq.freq
loadUserDict = hq.loadUserDict
addUserDict = hq.addUserDict
tradi2simp = hq._tradi2simp
strQ2B = hq._strQ2B

if __name__ == '__main__':
    huqie = Huqie(debug=True)
    # huqie.addUserDict("/tmp/tmp.new.tks.dict")
    # tks = huqie.qie(
    #     "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    # print(huqie.qieqie(tks))
    import time
    st = time.perf_counter()
    for _ in range(1000):
        tks = huqie.qie(
            "公开征求意见稿提出，境外投资者可使用自有人民币或外汇投资。使用外汇投资的，可通过债券持有人在香港人民币业务清算行及香港地区经批准可进入境内银行间外汇市场进行交易的境外人民币业务参加行（以下统称香港结算行）办理外汇资金兑换。香港结算行由此所产生的头寸可到境内银行间外汇市场平盘。使用外汇投资的，在其投资的债券到期或卖出后，原则上应兑换回外汇。")
        # print(huqie.qieqie(tks))
        huqie.qieqie(tks)
    print(f"cost {time.perf_counter()-st}s")
    # tks = huqie.qie(
    #     "多校划片就是一个小区对应多个小学初中，让买了学区房的家庭也不确定到底能上哪个学校。目的是通过这种方式为学区房降温，把就近入学落到实处。南京市长江大桥")
    # print(huqie.qieqie(tks))
    # tks = huqie.qie(
    #     "实际上当时他们已经将业务中心偏移到安全部门和针对政府企业的部门 Scripts are compiled and cached aaaaaaaaa")
    # print(huqie.qieqie(tks))
    # tks = huqie.qie("虽然我不怎么玩")
    # print(huqie.qieqie(tks))
    # tks = huqie.qie("蓝月亮如何在外资夹击中生存,那是全宇宙最有意思的")
    # print(huqie.qieqie(tks))
    # tks = huqie.qie(
    #     "涡轮增压发动机num最大功率,不像别的共享买车锁电子化的手段,我们接过来是否有意义,黄黄爱美食,不过，今天阿奇要讲到的这家农贸市场，说实话，还真蛮有特色的！不仅环境好，还打出了")
    # print(huqie.qieqie(tks))
    # tks = huqie.qie("这周日你去吗？这周日你有空吗？")
    # print(huqie.qieqie(tks))
    # tks = huqie.qie("Unity3D开发经验 测试开发工程师 c++双11双11 985 211 ")
    # print(huqie.qieqie(tks))
    # tks = huqie.qie(
    #     "数据分析项目经理|数据分析挖掘|数据分析方向|商品数据分析|搜索数据分析 sql python hive tableau Cocos2d-")
    # print(huqie.qieqie(tks))
    # if len(sys.argv) < 2:
    #     sys.exit()
    # huqie.DEBUG = False
    # huqie.loadUserDict(sys.argv[1])
    # of = open(sys.argv[2], "r")
    # while True:
    #     line = of.readline()
    #     if not line:
    #         break
    #     print(huqie.qie(line))
    # of.close()
