"""数据读取模块：负责将 CSV/TXT 数据转换为统一的 TextRecord 对象。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TextRecord:
    """一条文本样本。

    Attributes:
        record_id: 样本编号。
        text: 原始文本。
        label: 训练/测试标签；预测文件可为空。
    """

    record_id: str
    text: str
    label: Optional[str] = None


class DataReader:
    """读取不同格式的数据文件。"""

    def read_csv(
        self,
        file_path: str | Path,
        text_column: str = "text",
        label_column: str = "label",
        id_column: str = "id",
    ) -> list[TextRecord]:
        """读取 CSV 文件。

        CSV 至少应包含 text 列；若包含 label 列，可用于训练和测试。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")

        records: list[TextRecord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or text_column not in reader.fieldnames:
                raise ValueError(f"CSV 文件必须包含文本列：{text_column}")

            for index, row in enumerate(reader, start=1):
                text = (row.get(text_column) or "").strip()
                if not text:
                    continue
                record_id = (row.get(id_column) or str(index)).strip()
                label = (row.get(label_column) or "").strip() or None
                records.append(TextRecord(record_id=record_id, text=text, label=label))
        return records

    def read_txt(self, file_path: str | Path) -> list[TextRecord]:
        """读取纯文本文件，每行视为一条待预测文本。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")

        records: list[TextRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f, start=1):
                text = line.strip()
                if text:
                    records.append(TextRecord(record_id=str(index), text=text, label=None))
        return records
