"""智能文本情感分析系统单元测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analyzer import DataAnalyzer
from src.data_reader import DataReader
from src.model import NaiveBayesSentimentClassifier
from src.preprocessing import TextPreprocessor


class SentimentSystemTest(unittest.TestCase):
    def test_read_csv(self) -> None:
        reader = DataReader()
        records = reader.read_csv("sample_data/reviews.csv")
        self.assertGreaterEqual(len(records), 30)
        self.assertEqual(records[0].label, "positive")

    def test_tokenize_chinese_text(self) -> None:
        tokens = TextPreprocessor().tokenize("系统很好用")
        self.assertIn("系", tokens)
        self.assertIn("好_用", tokens)

    def test_model_can_predict(self) -> None:
        texts = ["体验很好 速度快", "非常卡顿 经常崩溃", "功能普通 基本可用"]
        labels = ["positive", "negative", "neutral"]
        model = NaiveBayesSentimentClassifier().fit(texts, labels)
        prediction = model.predict("速度快 很好用")
        self.assertIn(prediction, labels)

    def test_train_and_evaluate_exports_metrics(self) -> None:
        records = DataReader().read_csv("sample_data/reviews.csv")
        result = DataAnalyzer().train_and_evaluate(records, test_ratio=0.3, seed=42)
        self.assertIn("accuracy", result.metrics)
        self.assertEqual(result.metrics["test_count"], 9)

    def test_model_save_and_load(self) -> None:
        texts = ["很好用", "很糟糕"]
        labels = ["positive", "negative"]
        model = NaiveBayesSentimentClassifier().fit(texts, labels)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            model.save(path)
            loaded = NaiveBayesSentimentClassifier.load(path)
            self.assertEqual(loaded.predict("很好"), model.predict("很好"))


if __name__ == "__main__":
    unittest.main()
