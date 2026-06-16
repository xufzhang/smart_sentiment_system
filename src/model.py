"""核心算法模块：手写多项式朴素贝叶斯情感分类器。"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .preprocessing import TextPreprocessor


class NaiveBayesSentimentClassifier:
    """基于多项式朴素贝叶斯的三分类情感分析器。"""

    def __init__(self, alpha: float = 1.0, preprocessor: TextPreprocessor | None = None):
        if alpha <= 0:
            raise ValueError("alpha 必须大于 0")
        self.alpha = alpha
        self.preprocessor = preprocessor or TextPreprocessor()
        self.label_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.is_trained = False

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "NaiveBayesSentimentClassifier":
        """训练模型。"""
        self.label_counts.clear()
        self.token_counts.clear()
        self.total_tokens.clear()
        self.vocabulary.clear()

        sample_count = 0
        for text, label in zip(texts, labels):
            if not label:
                continue
            sample_count += 1
            self.label_counts[label] += 1
            tokens = self.preprocessor.tokenize(text)
            self.vocabulary.update(tokens)
            self.token_counts[label].update(tokens)
            self.total_tokens[label] += len(tokens)

        if sample_count == 0:
            raise ValueError("训练集不能为空，且必须包含 label")
        self.is_trained = True
        return self

    @property
    def labels(self) -> list[str]:
        return sorted(self.label_counts.keys())

    def _ensure_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

    def predict_log_proba(self, text: str) -> dict[str, float]:
        """返回各类别的对数概率。"""
        self._ensure_trained()
        tokens = self.preprocessor.tokenize(text)
        label_total = sum(self.label_counts.values())
        vocab_size = max(len(self.vocabulary), 1)
        scores: dict[str, float] = {}

        for label in self.labels:
            log_prob = math.log(self.label_counts[label] / label_total)
            denominator = self.total_tokens[label] + self.alpha * vocab_size
            for token in tokens:
                numerator = self.token_counts[label][token] + self.alpha
                log_prob += math.log(numerator / denominator)
            scores[label] = log_prob
        return scores

    def predict_proba(self, text: str) -> dict[str, float]:
        """返回归一化概率，便于展示置信度。"""
        log_scores = self.predict_log_proba(text)
        max_log = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        total = sum(exp_scores.values())
        return {label: value / total for label, value in exp_scores.items()}

    def predict(self, text: str) -> str:
        """预测单条文本类别。"""
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.get)

    def save(self, model_path: str | Path) -> None:
        """保存模型参数为 JSON。"""
        self._ensure_trained()
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "label_counts": dict(self.label_counts),
            "token_counts": {label: dict(counter) for label, counter in self.token_counts.items()},
            "total_tokens": dict(self.total_tokens),
            "vocabulary": sorted(self.vocabulary),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_path: str | Path) -> "NaiveBayesSentimentClassifier":
        """从 JSON 文件恢复模型。"""
        payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
        model = cls(alpha=payload["alpha"])
        model.label_counts = Counter(payload["label_counts"])
        model.token_counts = defaultdict(Counter, {k: Counter(v) for k, v in payload["token_counts"].items()})
        model.total_tokens = Counter(payload["total_tokens"])
        model.vocabulary = set(payload["vocabulary"])
        model.is_trained = True
        return model
