"""
interfaces.py — 抽象基类与数据类定义。

所有模块通过这些接口通信，主程序仅依赖抽象基类，
替换具体实现（如深度学习分割模型）时无需改动主流程。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================== 数据类 ========================

@dataclass
class FaceMeshResult:
    """人脸关键点检测结果。"""
    landmarks_3d: np.ndarray  # shape (478, 3)，MediaPipe 归一化坐标（含虹膜点）
    landmarks_px: np.ndarray  # shape (478, 2)，像素坐标 (x, y)
    left_iris_center_px: Tuple[int, int]   # 左眼虹膜中心像素坐标
    right_iris_center_px: Tuple[int, int]  # 右眼虹膜中心像素坐标
    image_shape: Tuple[int, int] = (0, 0)  # (H, W)


@dataclass
class HeadPoseResult:
    """公制头部姿态估计结果。"""
    rotation_matrix: np.ndarray   # 3x3 旋转矩阵
    translation_mm: np.ndarray    # (3,)，头部中心在相机坐标系下的毫米坐标
    scale_mm_per_unit: float = 1.0  # 从 MediaPipe 归一化坐标到毫米的缩放因子
    face_mesh_mm: np.ndarray = field(default_factory=lambda: np.zeros((478, 3)))
    # 公制面部网格 (478, 3)，单位毫米


@dataclass
class PtosisMetrics:
    """下垂检测指标。"""
    pfh_mm: float        # 平均睑裂高度（毫米）
    ipd_mm: float        # 瞳距（毫米）
    left_pfh_mm: float   # 左眼睑裂高度
    right_pfh_mm: float  # 右眼睑裂高度


@dataclass
class GazeResult:
    """眼球注视角度结果。"""
    left_yaw_deg: float = 0.0
    left_pitch_deg: float = 0.0
    right_yaw_deg: float = 0.0
    right_pitch_deg: float = 0.0
    avg_yaw_deg: float = 0.0
    avg_pitch_deg: float = 0.0


# ======================== 抽象基类 ========================

class FaceDetector(ABC):
    """人脸关键点检测器的抽象基类。"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[FaceMeshResult]:
        """
        从 BGR 图像中检测人脸，返回关键点和虹膜中心。

        Args:
            image: BGR 格式图像 (H, W, 3)。

        Returns:
            FaceMeshResult 或 None（未检测到人脸时）。
        """
        pass

    def release(self) -> None:
        """释放资源（子类可覆写）。"""
        pass


class HeadPoseMetricEstimator(ABC):
    """公制头部姿态估计器的抽象基类。"""

    @abstractmethod
    def estimate(
        self,
        face_mesh: FaceMeshResult,
        iris_diameter_mm: float = 11.8,
    ) -> HeadPoseResult:
        """
        利用虹膜直径先验恢复公制尺度，输出头部位姿。

        Args:
            face_mesh: 人脸关键点检测结果。
            iris_diameter_mm: 虹膜真实直径（毫米）。

        Returns:
            HeadPoseResult，包含旋转矩阵、毫米级平移和公制面部网格。
        """
        pass


class EyeTracker(ABC):
    """眼球追踪器的抽象基类。"""

    @abstractmethod
    def get_gaze_angles(
        self,
        face_mesh: FaceMeshResult,
        head_pose: HeadPoseResult,
    ) -> GazeResult:
        """
        计算眼球注视角度。

        Args:
            face_mesh: 人脸关键点检测结果。
            head_pose: 头部姿态估计结果。

        Returns:
            GazeResult，包含 yaw 和 pitch 角度（度）。
        """
        pass

    def set_reference(self, face_mesh: FaceMeshResult, head_pose: HeadPoseResult) -> None:
        """设置正面注视时的参考虹膜位置（校准时调用）。"""
        pass


class PtosisMetricCalculator(ABC):
    """下垂指标计算器的抽象基类。"""

    @abstractmethod
    def compute(
        self,
        head_pose: HeadPoseResult,
    ) -> PtosisMetrics:
        """
        根据公制面部网格计算 PFH (mm) 和 IPD (mm)。

        Args:
            head_pose: 包含公制面部网格的头部姿态结果。

        Returns:
            PtosisMetrics。
        """
        pass


class Personalizer(ABC):
    """个性化校准器的抽象基类。"""

    @abstractmethod
    def calibrate(self, calibration_data: List[Dict]) -> Dict:
        """
        使用校准数据优化个人参数。

        Args:
            calibration_data: 每个元素包含
                {head_pose, gaze_angles, iris_center_px, pfh_pixel, screen_point} 等。

        Returns:
            优化后的参数字典，如 {"eye_radius_scale": 1.05, "lid_offset_mm": 0.3}。
        """
        pass


class MuscleStrengthTester(ABC):
    """提上睑肌肌力测试器的抽象基类（Berke 法）。

    测试流程：
      1. start_test() 初始化，记录眉弓基线。
      2. 用户向下看 → capture_down_gaze() 记录下视上睑缘位置。
      3. 用户向上看 → capture_up_gaze() 记录上视上睑缘位置。
      4. compute_strength() 计算肌力 = 上视 - 下视的上睑缘垂直位移(mm)。
      5. get_grade() 返回分级。
    """

    @abstractmethod
    def start_test(self) -> None:
        """初始化测试，显示指导文本，重置内部状态。"""
        pass

    @abstractmethod
    def capture_down_gaze(
        self, landmarks_mm: np.ndarray, head_pose: HeadPoseResult
    ) -> None:
        """用户按下空格时调用，记录向下看时的上睑缘 3D 坐标。

        Args:
            landmarks_mm: 公制面部网格 (N, 3)，单位毫米。
            head_pose: 当前帧的头部姿态。

        Raises:
            ValueError: 检测到抬眉或头部移动过大时。
        """
        pass

    @abstractmethod
    def capture_up_gaze(
        self, landmarks_mm: np.ndarray, head_pose: HeadPoseResult
    ) -> None:
        """用户按下空格时调用，记录向上看时的上睑缘 3D 坐标。

        Args:
            landmarks_mm: 公制面部网格 (N, 3)，单位毫米。
            head_pose: 当前帧的头部姿态。

        Raises:
            ValueError: 检测到抬眉或头部移动过大时。
        """
        pass

    @abstractmethod
    def compute_strength(self) -> Optional[float]:
        """返回肌力值（mm）。如果两次记录不全则返回 None。"""
        pass

    @abstractmethod
    def get_grade(self, strength_mm: float) -> str:
        """根据肌力值返回分级字符串。

        Args:
            strength_mm: 肌力值（毫米）。

        Returns:
            分级："Good" / "Moderate" / "Poor"。
        """
        pass
