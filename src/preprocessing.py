"""文本预处理模块：清洗文本并提取适合中文短文本的字符 n-gram 特征。"""
from __future__ import annotations

import re


class TextPreprocessor:
    """将原始文本转换为机器学习模型可以使用的 token 列表。

    为提升小样本场景下的鲁棒性，除字符 n-gram 外，还加入少量
    可解释情感词典特征，例如 lex_positive、lex_negative。
    """

    _token_pattern = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+")
    _sentiment_lexicon = {
        "lex_positive": ["好", "优秀", "清晰", "准确", "稳定", "满意", "推荐", "顺手", "快", "及时"],
        "lex_negative": ["差", "慢", "卡顿", "崩溃", "乱码", "失败", "错误", "困难", "麻烦", "不准确"],
        "lex_neutral": ["一般", "普通", "基本", "尚可", "参考", "复核", "未测试", "还可以"],
    }

    def clean(self, text: str) -> str:
        """执行基础清洗：统一大小写、去除 URL、压缩空白。"""
        text = text.lower().strip()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str) -> list[str]:
        """提取 unigram 与 bigram 特征。

        中文没有天然空格分词，本实验使用字符级 unigram + 相邻 bigram，
        能在不依赖外部分词库的情况下捕捉“好用”“卡顿”等短语特征。
        """
        cleaned = self.clean(text)
        base_tokens = self._token_pattern.findall(cleaned)
        if not base_tokens:
            return []
        bigrams = [f"{base_tokens[i]}_{base_tokens[i + 1]}" for i in range(len(base_tokens) - 1)]
        lexicon_tokens = []
        for feature_name, words in self._sentiment_lexicon.items():
            for word in words:
                if word in cleaned:
                    # 追加两次以增强少量关键情感词的权重。
                    lexicon_tokens.extend([feature_name, feature_name])
                    break
        return base_tokens + bigrams + lexicon_tokens
