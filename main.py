"""智能文本情感分析系统入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.analyzer import DataAnalyzer
from src.data_reader import DataReader, TextRecord
from src.exporter import ResultExporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="智能文本情感分析系统")
    parser.add_argument("--data", default="sample_data/reviews.csv", help="训练/测试 CSV 数据路径")
    parser.add_argument("--output", default="output", help="结果输出目录")
    parser.add_argument("--test-ratio", type=float, default=0.3, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--predict", nargs="*", help="训练后额外预测的文本，可输入多条")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reader = DataReader()
    records = reader.read_csv(args.data)

    analyzer = DataAnalyzer()
    result = analyzer.train_and_evaluate(records, test_ratio=args.test_ratio, seed=args.seed)

    # 对用户通过命令行输入的新文本进行预测，并追加到展示结果中。
    extra_predictions = []
    if args.predict:
        extra_records = [TextRecord(record_id=f"new-{i}", text=text) for i, text in enumerate(args.predict, start=1)]
        extra_predictions = analyzer.predict_records(extra_records)

    exporter = ResultExporter(args.output)
    exported = exporter.export_all(result.predictions + extra_predictions, result.metrics)
    analyzer.classifier.save(Path(args.output) / "sentiment_model.json")

    print("智能文本情感分析系统运行完成")
    print(f"训练样本数：{result.metrics['train_count']}，测试样本数：{result.metrics['test_count']}")
    print(f"准确率：{result.metrics['accuracy']:.2%}，宏平均 F1：{result.metrics['macro_f1']:.2%}")
    if extra_predictions:
        print("\n新增文本预测：")
        for item in extra_predictions:
            print(f"- {item['text']} => {item['predicted_label']}（置信度 {item['confidence']:.2%}）")
    print("\n已输出文件：")
    for name, path in exported.items():
        print(f"- {name}: {path}")
    print(f"- model: {Path(args.output) / 'sentiment_model.json'}")


if __name__ == "__main__":
    main()
