"""
berke_muscle_tester.py — 提上睑肌肌力测量（Berke 法）实现。

Berke 法流程：
  1. 固定额肌（通过检测眉弓位移验证用户未抬眉）。
  2. 用户向下看，记录上睑缘垂直位置 (mm)。
  3. 用户向上看，记录上睑缘垂直位置 (mm)。
  4. 肌力 = 上视位置 - 下视位置 (mm)。
  5. 分级：良好 ≥8mm，中等 4-7mm，差 ≤4mm。
"""

from enum import Enum, auto
from typing import Optional

import numpy as np

from .. import config
from ..interfaces import HeadPoseResult, MuscleStrengthTester


class TestState(Enum):
    """肌力测试状态机。"""
    IDLE = auto()
    WAITING_DOWN = auto()   # 等待用户向下看并按空格
    WAITING_UP = auto()     # 等待用户向上看并按空格
    COMPLETED = auto()


class BerkeMuscleTester(MuscleStrengthTester):
    """基于 Berke 法的提上睑肌肌力测试器。

    通过比较向下看和向上看时上睑缘的垂直位移（毫米）来评估肌力。
    同时监测眉弓位移以验证额肌未参与（模拟医生按压眉弓）。

    Attributes:
        state: 当前测试状态。
        down_lid_y_mm: 向下看时上睑缘的平均 y 坐标 (mm)。
        up_lid_y_mm: 向上看时上睑缘的平均 y 坐标 (mm)。
        warning: 最近一次操作的警告信息（无警告时为空字符串）。
    """

    def __init__(self) -> None:
        self.state: TestState = TestState.IDLE
        self.down_lid_y_mm: Optional[float] = None
        self.up_lid_y_mm: Optional[float] = None
        self._brow_baseline_y_mm: Optional[float] = None
        self._head_baseline_mm: Optional[np.ndarray] = None
        self.warning: str = ""

    @property
    def prompt_text(self) -> str:
        """返回当前状态对应的用户提示文本。"""
        prompts = {
            TestState.IDLE: "",
            TestState.WAITING_DOWN: "MUSCLE TEST: Look DOWN, then press SPACE",
            TestState.WAITING_UP: "MUSCLE TEST: Look UP, then press SPACE",
            TestState.COMPLETED: "",
        }
        return prompts.get(self.state, "")

    def start_test(self) -> None:
        """初始化测试，重置内部状态。"""
        self.state = TestState.WAITING_DOWN
        self.down_lid_y_mm = None
        self.up_lid_y_mm = None
        self._brow_baseline_y_mm = None
        self._head_baseline_mm = None
        self.warning = ""
        print("[MUSCLE] 肌力测试开始。请向下看，然后按空格键。")

    def capture_down_gaze(
        self, landmarks_mm: np.ndarray, head_pose: HeadPoseResult
    ) -> None:
        """记录向下看时的上睑缘位置，并设置眉弓和头部基线。

        Args:
            landmarks_mm: 公制面部网格 (N, 3)，单位毫米。
            head_pose: 当前帧头部姿态。

        Raises:
            ValueError: 检测条件不满足时。
        """
        self.warning = ""

        # 设置眉弓基线（首次捕获时）
        self._brow_baseline_y_mm = self._get_brow_y_mm(landmarks_mm)
        self._head_baseline_mm = head_pose.translation_mm.copy()

        # 记录上睑缘 y 坐标
        self.down_lid_y_mm = self._get_upper_lid_y_mm(landmarks_mm)
        self.state = TestState.WAITING_UP
        print(f"[MUSCLE] 下视位置记录: {self.down_lid_y_mm:.2f} mm。请向上看，然后按空格键。")

    def capture_up_gaze(
        self, landmarks_mm: np.ndarray, head_pose: HeadPoseResult
    ) -> None:
        """记录向上看时的上睑缘位置，并验证额肌和头部稳定性。

        Args:
            landmarks_mm: 公制面部网格 (N, 3)，单位毫米。
            head_pose: 当前帧头部姿态。

        Raises:
            ValueError: 检测到抬眉或头部移动过大时。
        """
        self.warning = ""

        # 额肌阻断验证
        brow_warning = self._check_brow_rise(landmarks_mm)
        if brow_warning:
            self.warning = brow_warning
            raise ValueError(brow_warning)

        # 头部稳定性验证
        head_warning = self._check_head_movement(head_pose)
        if head_warning:
            self.warning = head_warning
            raise ValueError(head_warning)

        # 记录上睑缘 y 坐标
        self.up_lid_y_mm = self._get_upper_lid_y_mm(landmarks_mm)
        self.state = TestState.COMPLETED
        print(f"[MUSCLE] 上视位置记录: {self.up_lid_y_mm:.2f} mm。测试完成。")

    def compute_strength(self) -> Optional[float]:
        """计算肌力值（mm）。

        肌力 = |下视上睑缘 y - 上视上睑缘 y|。
        在 MediaPipe 坐标系中 y 轴向下为正，所以向下看时 y 值更大，
        向上看时 y 值更小，肌力 = down_y - up_y。

        Returns:
            肌力值（毫米），两次记录不全时返回 None。
        """
        if self.down_lid_y_mm is None or self.up_lid_y_mm is None:
            return None
        # 向下看时上睑缘 y 较大，向上看时较小
        strength = abs(self.down_lid_y_mm - self.up_lid_y_mm)
        return round(strength, 2)

    def get_grade(self, strength_mm: float) -> str:
        """根据肌力值返回分级。

        Args:
            strength_mm: 肌力值（毫米）。

        Returns:
            "Good" (≥8mm) / "Moderate" (4-7mm) / "Poor" (≤4mm)。
        """
        if strength_mm >= config.STRENGTH_GOOD:
            return "Good"
        elif strength_mm >= config.STRENGTH_MODERATE_LOW:
            return "Moderate"
        else:
            return "Poor"

    @staticmethod
    def _get_upper_lid_y_mm(landmarks_mm: np.ndarray) -> float:
        """获取左右眼上睑缘 y 坐标的平均值（毫米）。

        Args:
            landmarks_mm: 公制面部网格 (N, 3)。

        Returns:
            上睑缘平均 y 坐标 (mm)。
        """
        left_y = landmarks_mm[config.LEFT_UPPER_LID_MARGIN][1]
        right_y = landmarks_mm[config.RIGHT_UPPER_LID_MARGIN][1]
        return float((left_y + right_y) / 2.0)

    @staticmethod
    def _get_brow_y_mm(landmarks_mm: np.ndarray) -> float:
        """获取左右眉弓 y 坐标的平均值（毫米）。

        Args:
            landmarks_mm: 公制面部网格 (N, 3)。

        Returns:
            眉弓平均 y 坐标 (mm)。
        """
        left_y = landmarks_mm[config.LEFT_BROW_RIDGE][1]
        right_y = landmarks_mm[config.RIGHT_BROW_RIDGE][1]
        return float((left_y + right_y) / 2.0)

    def _check_brow_rise(self, landmarks_mm: np.ndarray) -> str:
        """检测是否抬眉。

        Args:
            landmarks_mm: 当前帧公制面部网格。

        Returns:
            警告信息字符串（无问题时返回空字符串）。
        """
        if self._brow_baseline_y_mm is None:
            return ""

        current_brow_y = self._get_brow_y_mm(landmarks_mm)
        delta = abs(current_brow_y - self._brow_baseline_y_mm)

        if delta > config.BROW_RISE_THRESHOLD_MM:
            return f"Brow rise detected ({delta:.1f} mm)! Hold brow still and retry."
        return ""

    def _check_head_movement(self, head_pose: HeadPoseResult) -> str:
        """检测头部是否移动过大。

        Args:
            head_pose: 当前帧头部姿态。

        Returns:
            警告信息字符串（无问题时返回空字符串）。
        """
        if self._head_baseline_mm is None:
            return ""

        delta = np.linalg.norm(head_pose.translation_mm - self._head_baseline_mm)

        if delta > config.HEAD_MOVEMENT_THRESHOLD_MM:
            return f"Head movement too large ({delta:.1f} mm)! Keep head still and retry."
        return ""
