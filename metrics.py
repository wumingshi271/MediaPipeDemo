"""
下垂指标计算模块。

扩展点:
  - 添加新指标（如 MRD-1、MRD-2、眉眼距等）只需在 PtosisMetricsCalculator 中
    新增 calculate_xxx() 方法，并在 evaluate() 中调用。
  - 要整体替换计算逻辑，可继承 BaseMetricsCalculator。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


class BaseMetricsCalculator(ABC):
    """指标计算器的抽象基类。"""

    @abstractmethod
    def evaluate(
        self, landmarks: List[Tuple[int, int]]
    ) -> Dict[str, Optional[float]]:
        """
        根据关键点坐标计算所有指标。

        Args:
            landmarks: 像素坐标列表 [(x, y), ...]。

        Returns:
            指标字典，例如 {"left_pfh": 12.3, "right_pfh": 15.1}。
            某项计算失败时对应值为 None。
        """
        ...


class PtosisMetricsCalculator(BaseMetricsCalculator):
    """基于几何距离的下垂指标计算器。"""

    def __init__(
        self,
        left_lid_pairs: List[Tuple[int, int]] = None,
        right_lid_pairs: List[Tuple[int, int]] = None,
        threshold: float = None,
    ) -> None:
        self.left_lid_pairs = left_lid_pairs or config.LEFT_EYE_LID_PAIRS
        self.right_lid_pairs = right_lid_pairs or config.RIGHT_EYE_LID_PAIRS
        self.threshold = threshold or config.PFH_PTOSIS_THRESHOLD

    @staticmethod
    def _calculate_pfh(
        landmarks: List[Tuple[int, int]],
        lid_pairs: List[Tuple[int, int]],
    ) -> Optional[float]:
        """计算一只眼的睑裂高度 (PFH)，取多对上下睑点的平均垂直距离。"""
        distances = []
        for upper_idx, lower_idx in lid_pairs:
            if upper_idx >= len(landmarks) or lower_idx >= len(landmarks):
                continue
            upper = np.array(landmarks[upper_idx])
            lower = np.array(landmarks[lower_idx])
            dist = float(np.linalg.norm(upper - lower))
            distances.append(dist)

        if not distances:
            return None
        return float(np.mean(distances))

    def evaluate(
        self, landmarks: List[Tuple[int, int]]
    ) -> Dict[str, Optional[float]]:
        left_pfh = self._calculate_pfh(landmarks, self.left_lid_pairs)
        right_pfh = self._calculate_pfh(landmarks, self.right_lid_pairs)

        return {
            "left_pfh": round(left_pfh, 1) if left_pfh is not None else None,
            "right_pfh": round(right_pfh, 1) if right_pfh is not None else None,
            "left_ptosis": left_pfh is not None and left_pfh < self.threshold,
            "right_ptosis": right_pfh is not None and right_pfh < self.threshold,
        }

    # ---- 扩展示例 ----
    # def calculate_mrd1(self, landmarks, ...) -> Optional[float]:
    #     """计算 MRD-1（瞳孔中心到上睑缘的距离）。"""
    #     pass
