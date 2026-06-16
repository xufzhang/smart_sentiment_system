"""结果导出模块：将预测结果和评价指标保存为文件。"""
from __future__ import annotations

import csv
import json
from pathlib import Path


class ResultExporter:
    """导出预测明细、指标和混淆矩阵。"""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_predictions(self, predictions: list[dict[str, str | float]]) -> Path:
        path = self.output_dir / "predictions.csv"
        fieldnames = ["id", "text", "true_label", "predicted_label", "confidence"]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)
        return path

    def export_metrics(self, metrics: dict[str, object]) -> Path:
        path = self.output_dir / "metrics.json"
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def export_confusion_matrix(self, metrics: dict[str, object]) -> Path:
        labels = list(metrics.get("labels", []))
        matrix = metrics.get("confusion_matrix", {})
        path = self.output_dir / "confusion_matrix.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["actual/predicted", *labels])
            for actual in labels:
                row = [actual]
                row.extend(matrix.get(actual, {}).get(predicted, 0) for predicted in labels)
                writer.writerow(row)
        return path

    def export_summary(self, metrics: dict[str, object]) -> Path:
        path = self.output_dir / "summary.txt"
        lines = [
            "智能文本情感分析系统运行摘要",
            "=" * 24,
            f"训练样本数：{metrics.get('train_count', 'N/A')}",
            f"测试样本数：{metrics.get('test_count', 'N/A')}",
            f"准确率：{metrics.get('accuracy', 0):.2%}",
            f"宏平均 F1：{metrics.get('macro_f1', 0):.2%}",
            "",
            "类别指标：",
        ]
        per_label = metrics.get("per_label", {})
        for label, score in per_label.items():
            lines.append(
                f"- {label}: Precision={score['precision']:.2f}, "
                f"Recall={score['recall']:.2f}, F1={score['f1']:.2f}, Support={int(score['support'])}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def export_all(self, predictions: list[dict[str, str | float]], metrics: dict[str, object]) -> dict[str, Path]:
        return {
            "predictions": self.export_predictions(predictions),
            "metrics": self.export_metrics(metrics),
            "confusion_matrix": self.export_confusion_matrix(metrics),
            "summary": self.export_summary(metrics),
        }
