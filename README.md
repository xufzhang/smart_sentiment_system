# 智能文本情感分析系统

本项目是一个面向课程实验的简单人工智能应用系统。系统使用面向对象设计实现数据输入、核心算法、结果导出和单元测试，核心算法为“字符 n-gram + 情感词典增强”的手写多项式朴素贝叶斯文本分类器。

## 目录结构

```text
smart_sentiment_system/
├── main.py                      # 程序入口
├── src/                         # 系统源代码
│   ├── data_reader.py            # 数据读取模块
│   ├── preprocessing.py          # 文本清洗与特征提取模块
│   ├── model.py                  # 朴素贝叶斯分类模型
│   ├── analyzer.py               # 训练、预测、评价流程
│   └── exporter.py               # 结果导出模块
├── sample_data/reviews.csv       # 样例数据集
├── tests/test_sentiment.py       # 单元测试
├── output/                       # 运行结果目录
└── docs/                         # 文档与报告
```

## 运行环境

- Python 3.10 及以上
- VSCode 或 TRAE
- Git
- 无强制第三方依赖，主体程序仅使用 Python 标准库

## 快速运行

```bash
python main.py --data sample_data/reviews.csv --output output --predict "这个系统很好用" "程序经常崩溃"
```

运行后会在 `output` 目录生成：

- `predictions.csv`：预测明细
- `metrics.json`：评价指标
- `confusion_matrix.csv`：混淆矩阵
- `summary.txt`：文本摘要
- `sentiment_model.json`：训练后的模型参数

## 单元测试

```bash
python -m unittest discover -s tests
```

## GitHub/Gitee URL 占位

- GitHub: `https://github.com/your-name/smart-sentiment-system`
- Gitee: `https://gitee.com/your-name/smart-sentiment-system`
