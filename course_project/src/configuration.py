"""Nạp và truy xuất cấu hình YAML chuẩn của dự án."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ProjectConfiguration:
    """Đọc cấu hình dự án và cung cấp truy cập có kiểm tra tới các nhóm tham số."""

    DEFAULT_PATH = Path("configs/default.yaml")

    def __init__(self, config_path: str | Path = DEFAULT_PATH) -> None:
        """Nạp cấu hình YAML từ đường dẫn được chỉ định.

        Args:
            config_path: Đường dẫn tệp cấu hình; mặc định là ``configs/default.yaml``.

        Raises:
            FileNotFoundError: Nếu tệp cấu hình không tồn tại.
            ValueError: Nếu nội dung YAML rỗng hoặc không phải ánh xạ.
        """
        self.path = Path(config_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình: {self.path}")

        with self.path.open("r", encoding="utf-8") as config_file:
            loaded: Any = yaml.safe_load(config_file)
        if not isinstance(loaded, dict):
            raise ValueError(f"Cấu hình phải là ánh xạ YAML: {self.path}")
        self._values: Dict[str, Any] = loaded

    def section(self, name: str, required: bool = True) -> Dict[str, Any]:
        """Lấy một nhóm cấu hình dưới dạng từ điển.

        Args:
            name: Tên nhóm cấu hình ở cấp cao nhất.
            required: Nếu đúng, báo lỗi khi nhóm không tồn tại.

        Returns:
            Từ điển tham số thuộc nhóm được yêu cầu.

        Raises:
            KeyError: Nếu nhóm bắt buộc không tồn tại.
            TypeError: Nếu giá trị nhóm không phải từ điển.
        """
        value = self._values.get(name)
        if value is None and not required:
            return {}
        if value is None:
            raise KeyError(f"Thiếu nhóm cấu hình bắt buộc: {name}")
        if not isinstance(value, dict):
            raise TypeError(f"Nhóm cấu hình '{name}' phải là ánh xạ YAML.")
        return value

    def get(self, section: str, key: str, default: Optional[Any] = None) -> Any:
        """Lấy một tham số trong nhóm cấu hình.

        Args:
            section: Tên nhóm cấu hình.
            key: Tên tham số trong nhóm.
            default: Giá trị thay thế nếu tham số không tồn tại.

        Returns:
            Giá trị cấu hình hoặc ``default``.
        """
        return self.section(section, required=False).get(key, default)