"""业务分析模块：组织训练、预测、评价流程。"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .data_reader import TextRecord
from .model import NaiveBayesSentimentClassifier


@dataclass
class AnalysisResult:
    """一次训练评估任务的输出。"""

    predictions: list[dict[str, str | float]]
    metrics: dict[str, object]


class DataAnalyzer:
    """情感分析任务编排器，负责划分数据、训练模型和生成评价指标。"""

    def __init__(self, classifier: NaiveBayesSentimentClassifier | None = None):
        self.classifier = classifier or NaiveBayesSentimentClassifier()

    def split_data(
        self,
        records: list[TextRecord],
        test_ratio: float = 0.3,
        seed: int = 42,
    ) -> tuple[list[TextRecord], list[TextRecord]]:
        labeled = [record for record in records if record.label]
        if len(labeled) < 3:
            raise ValueError("至少需要 3 条带标签样本")
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio 必须在 0 和 1 之间")

        shuffled = labeled[:]
        random.Random(seed).shuffle(shuffled)
        test_size = max(1, int(len(shuffled) * test_ratio))
        return shuffled[test_size:], shuffled[:test_size]

    def train(self, train_records: list[TextRecord]) -> None:
        texts = [record.text for record in train_records]
        labels = [record.label or "" for record in train_records]
        self.classifier.fit(texts, labels)

    def predict_records(self, records: list[TextRecord]) -> list[dict[str, str | float]]:
        output: list[dict[str, str | float]] = []
        for record in records:
            probabilities = self.classifier.predict_proba(record.text)
            predicted_label = max(probabilities, key=probabilities.get)
            output.append(
                {
                    "id": record.record_id,
                    "text": record.text,
                    "true_label": record.label or "",
                    "predicted_label": predicted_label,
                    "confidence": round(probabilities[predicted_label], 4),
                }
            )
        return output

    def evaluate(self, predictions: list[dict[str, str | float]]) -> dict[str, object]:
        true_labels = [str(item["true_label"]) for item in predictions if item["true_label"]]
        predicted_labels = [str(item["predicted_label"]) for item in predictions if item["true_label"]]
        labels = sorted(set(true_labels) | set(predicted_labels))
        total = len(true_labels)
        correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
        accuracy = correct / total if total else 0.0

        per_label: dict[str, dict[str, float]] = {}
        for label in labels:
            tp = sum(t == label and p == label for t, p in zip(true_labels, predicted_labels))
            fp = sum(t != label and p == label for t, p in zip(true_labels, predicted_labels))
            fn = sum(t == label and p != label for t, p in zip(true_labels, predicted_labels))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_label[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": float(Counter(true_labels)[label]),
            }

        confusion_matrix = {
            actual: {predicted: 0 for predicted in labels}
            for actual in labels
        }
        for actual, predicted in zip(true_labels, predicted_labels):
            confusion_matrix[actual][predicted] += 1

        macro_f1 = sum(score["f1"] for score in per_label.values()) / len(labels) if labels else 0.0
        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "labels": labels,
            "per_label": per_label,
            "confusion_matrix": confusion_matrix,
            "sample_count": total,
        }

    def train_and_evaluate(self, records: list[TextRecord], test_ratio: float = 0.3, seed: int = 42) -> AnalysisResult:
        train_records, test_records = self.split_data(records, test_ratio=test_ratio, seed=seed)
        self.train(train_records)
        predictions = self.predict_records(test_records)
        metrics = self.evaluate(predictions)
        metrics["train_count"] = len(train_records)
        metrics["test_count"] = len(test_records)
        return AnalysisResult(predictions=predictions, metrics=metrics)
