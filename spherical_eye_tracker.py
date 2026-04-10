"""
spherical_eye_tracker.py — 基于球形眼球模型的眼球角度解算。

参考：Cristina & Camilleri 2016。
原理：
  - 眼球近似为半径 R_eye_mm 的球体。
  - 正面注视时虹膜中心位置为参考点。
  - 当前帧虹膜中心相对于参考点的像素位移 (dx, dy)，
    转换为毫米后投影到球面上，得到偏航角(yaw)和俯仰角(pitch)。
  - 最终眼球注视角度 = 眼球相对头部角度 + 头部欧拉角。
"""

from typing import Optional, Tuple

import numpy as np

import config
from interfaces import EyeTracker, FaceMeshResult, GazeResult, HeadPoseResult


class SphericalEyeTracker(EyeTracker):
    """基于球形眼球模型的眼球追踪器。

    Attributes:
        eye_radius_mm: 眼球半径（毫米），可通过校准更新。
        _ref_left_iris_px: 参考帧左眼虹膜中心像素坐标。
        _ref_right_iris_px: 参考帧右眼虹膜中心像素坐标。
    """

    def __init__(self, eye_radius_mm: float = config.EYE_RADIUS_MM) -> None:
        self.eye_radius_mm = eye_radius_mm
        self._ref_left_iris_px: Optional[np.ndarray] = None
        self._ref_right_iris_px: Optional[np.ndarray] = None
        self._ref_scale: float = 1.0  # mm/pixel at reference frame

    def set_reference(self, face_mesh: FaceMeshResult, head_pose: HeadPoseResult) -> None:
        """记录正面注视时的虹膜中心位置和尺度。

        应在校准开始时（用户正视摄像头）调用。

        Args:
            face_mesh: 正面注视时的检测结果。
            head_pose: 正面注视时的头部姿态。
        """
        self._ref_left_iris_px = np.array(face_mesh.left_iris_center_px, dtype=np.float64)
        self._ref_right_iris_px = np.array(face_mesh.right_iris_center_px, dtype=np.float64)
        self._ref_scale = head_pose.scale_mm_per_unit

    def get_gaze_angles(
        self,
        face_mesh: FaceMeshResult,
        head_pose: HeadPoseResult,
    ) -> GazeResult:
        """计算眼球注视角度。

        Args:
            face_mesh: 当前帧人脸检测结果。
            head_pose: 当前帧头部姿态。

        Returns:
            GazeResult，包含左右眼及平均的 yaw/pitch（度）。
        """
        # 如果没有参考帧，用当前帧初始化
        if self._ref_left_iris_px is None:
            self.set_reference(face_mesh, head_pose)

        # 像素到毫米的转换因子
        # 使用虹膜像素直径计算: mm_per_px = iris_diameter_mm / iris_pixel_diameter
        iris_px_diam = self._compute_iris_pixel_diameter(face_mesh.landmarks_px)
        if iris_px_diam > 1e-6:
            mm_per_px = config.IRIS_DIAMETER_MM / iris_px_diam
        else:
            mm_per_px = self._ref_scale / face_mesh.image_shape[1] if face_mesh.image_shape[1] > 0 else 0.01

        # 左眼
        left_iris = np.array(face_mesh.left_iris_center_px, dtype=np.float64)
        left_dx_px = left_iris[0] - self._ref_left_iris_px[0]
        left_dy_px = left_iris[1] - self._ref_left_iris_px[1]
        left_yaw, left_pitch = self._pixel_to_angle(left_dx_px, left_dy_px, mm_per_px)

        # 右眼
        right_iris = np.array(face_mesh.right_iris_center_px, dtype=np.float64)
        right_dx_px = right_iris[0] - self._ref_right_iris_px[0]
        right_dy_px = right_iris[1] - self._ref_right_iris_px[1]
        right_yaw, right_pitch = self._pixel_to_angle(right_dx_px, right_dy_px, mm_per_px)

        # 补偿头部姿态
        head_yaw, head_pitch, _ = self._rotation_to_euler(head_pose.rotation_matrix)

        left_yaw += head_yaw
        left_pitch += head_pitch
        right_yaw += head_yaw
        right_pitch += head_pitch

        avg_yaw = (left_yaw + right_yaw) / 2.0
        avg_pitch = (left_pitch + right_pitch) / 2.0

        return GazeResult(
            left_yaw_deg=left_yaw,
            left_pitch_deg=left_pitch,
            right_yaw_deg=right_yaw,
            right_pitch_deg=right_pitch,
            avg_yaw_deg=avg_yaw,
            avg_pitch_deg=avg_pitch,
        )

    def _pixel_to_angle(
        self, dx_px: float, dy_px: float, mm_per_px: float
    ) -> Tuple[float, float]:
        """将像素位移转换为球面角度。

        Args:
            dx_px: 水平像素位移。
            dy_px: 垂直像素位移。
            mm_per_px: 毫米/像素比。

        Returns:
            (yaw_deg, pitch_deg)。
        """
        dx_mm = dx_px * mm_per_px
        dy_mm = dy_px * mm_per_px

        # 投影到球面：angle = arcsin(displacement / radius)
        # 限制在 [-1, 1] 以避免 arcsin 域错误
        sin_yaw = np.clip(dx_mm / self.eye_radius_mm, -1.0, 1.0)
        sin_pitch = np.clip(dy_mm / self.eye_radius_mm, -1.0, 1.0)

        yaw_deg = float(np.degrees(np.arcsin(sin_yaw)))
        pitch_deg = float(np.degrees(np.arcsin(sin_pitch)))

        return yaw_deg, pitch_deg

    @staticmethod
    def _compute_iris_pixel_diameter(landmarks_px: np.ndarray) -> float:
        """计算虹膜像素直径（取左右眼平均）。"""
        diameters = []
        for edge_indices in [config.LEFT_IRIS_EDGE_INDICES, config.RIGHT_IRIS_EDGE_INDICES]:
            if max(edge_indices) >= len(landmarks_px):
                continue
            pts = landmarks_px[edge_indices].astype(np.float64)
            d1 = np.linalg.norm(pts[0] - pts[2])
            d2 = np.linalg.norm(pts[1] - pts[3])
            diameters.append((d1 + d2) / 2.0)
        return float(np.mean(diameters)) if diameters else 0.0

    @staticmethod
    def _rotation_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """从旋转矩阵提取欧拉角 (yaw, pitch, roll)（度）。

        使用 ZYX 顺序。
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        else:
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            roll = 0.0
        return yaw, pitch, roll
