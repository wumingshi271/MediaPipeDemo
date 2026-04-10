"""
procrustes_headpose.py — 公制头部姿态恢复（简化版）。

简化策略（如 need.md 4.1 节允许）：
  1. 通过虹膜像素直径与已知虹膜真实直径(11.8mm)的比值，估算尺度因子。
  2. 将 MediaPipe 归一化 3D 坐标乘以尺度因子，得到公制网格(mm)。
  3. 旋转矩阵从 MediaPipe 的 facial_transformation_matrixes 提取；
     若不可用，则通过关键点 PnP 求解。
  4. 深度(translation z)从虹膜尺度反推。

注意：这是简化版实现。如需更高精度，可替换为完整的径向 Procrustes 方法。
"""

import numpy as np

import config
from interfaces import FaceMeshResult, HeadPoseMetricEstimator, HeadPoseResult


class SimplifiedHeadPoseEstimator(HeadPoseMetricEstimator):
    """基于虹膜尺度先验的简化公制头部姿态估计器。

    使用虹膜直径将 MediaPipe 归一化坐标转换为毫米单位，
    并利用 MediaPipe 自带的变换矩阵或简单几何推算头部位姿。
    """

    def __init__(self) -> None:
        self._initial_scale: float = 0.0  # 首帧锁定的尺度因子
        self._initialized = False

    def estimate(
        self,
        face_mesh: FaceMeshResult,
        iris_diameter_mm: float = config.IRIS_DIAMETER_MM,
    ) -> HeadPoseResult:
        """估计公制头部姿态。

        Args:
            face_mesh: 人脸关键点检测结果（含虹膜点）。
            iris_diameter_mm: 虹膜真实直径（毫米），默认 11.8mm。

        Returns:
            HeadPoseResult，包含旋转矩阵、毫米平移和公制面部网格。
        """
        landmarks_3d = face_mesh.landmarks_3d  # (478, 3) 归一化坐标
        h, w = face_mesh.image_shape

        # 1. 计算虹膜像素直径
        iris_pixel_diam = self._compute_iris_pixel_diameter(face_mesh.landmarks_px)

        # 2. 计算尺度因子：mm / 归一化单位
        #    虹膜像素直径对应的归一化宽度
        iris_norm_diam = self._compute_iris_norm_diameter(landmarks_3d)

        if iris_norm_diam > 1e-6:
            scale_mm_per_unit = iris_diameter_mm / iris_norm_diam
        elif self._initialized:
            scale_mm_per_unit = self._initial_scale
        else:
            # 回退到默认深度假设
            scale_mm_per_unit = config.DEFAULT_DEPTH_MM / 1.0  # 粗略估计
            print("[WARNING] 无法检测虹膜尺寸，使用默认深度假设")

        if not self._initialized:
            self._initial_scale = scale_mm_per_unit
            self._initialized = True

        # 3. 构建公制面部网格
        face_mesh_mm = landmarks_3d.copy() * scale_mm_per_unit

        # 4. 估算旋转矩阵（简化：从关键点计算面部朝向）
        rotation_matrix = self._estimate_rotation(landmarks_3d)

        # 5. 估算平移（毫米）
        #    使用鼻尖(idx=1)作为面部中心
        nose_tip_mm = face_mesh_mm[1]
        # z 深度从虹膜尺度反推：depth = (iris_real * focal) / iris_pixel
        # 简化：focal ≈ w（像素），depth_mm = iris_diameter_mm * w / iris_pixel_diam
        if iris_pixel_diam > 1e-6:
            depth_mm = iris_diameter_mm * w / iris_pixel_diam
        else:
            depth_mm = config.DEFAULT_DEPTH_MM

        translation_mm = np.array([
            nose_tip_mm[0] - face_mesh_mm[:, 0].mean(),
            nose_tip_mm[1] - face_mesh_mm[:, 1].mean(),
            depth_mm,
        ])

        return HeadPoseResult(
            rotation_matrix=rotation_matrix,
            translation_mm=translation_mm,
            scale_mm_per_unit=scale_mm_per_unit,
            face_mesh_mm=face_mesh_mm,
        )

    @staticmethod
    def _compute_iris_pixel_diameter(landmarks_px: np.ndarray) -> float:
        """计算虹膜像素直径（取左右眼平均）。

        Args:
            landmarks_px: 像素坐标数组 (N, 2)。

        Returns:
            虹膜像素直径。
        """
        diameters = []
        for edge_indices in [config.LEFT_IRIS_EDGE_INDICES, config.RIGHT_IRIS_EDGE_INDICES]:
            if max(edge_indices) >= len(landmarks_px):
                continue
            pts = landmarks_px[edge_indices].astype(np.float64)
            # 两对对角点的距离取平均
            d1 = np.linalg.norm(pts[0] - pts[2])
            d2 = np.linalg.norm(pts[1] - pts[3])
            diameters.append((d1 + d2) / 2.0)

        return float(np.mean(diameters)) if diameters else 0.0

    @staticmethod
    def _compute_iris_norm_diameter(landmarks_3d: np.ndarray) -> float:
        """计算虹膜在归一化 3D 坐标中的直径。"""
        diameters = []
        for edge_indices in [config.LEFT_IRIS_EDGE_INDICES, config.RIGHT_IRIS_EDGE_INDICES]:
            if max(edge_indices) >= len(landmarks_3d):
                continue
            pts = landmarks_3d[edge_indices]
            d1 = np.linalg.norm(pts[0] - pts[2])
            d2 = np.linalg.norm(pts[1] - pts[3])
            diameters.append((d1 + d2) / 2.0)

        return float(np.mean(diameters)) if diameters else 0.0

    @staticmethod
    def _estimate_rotation(landmarks_3d: np.ndarray) -> np.ndarray:
        """从关键点估计面部旋转矩阵。

        使用鼻梁方向和左右眼连线构建面部坐标系。

        Returns:
            3x3 旋转矩阵。
        """
        # 关键点
        nose_tip = landmarks_3d[1]       # 鼻尖
        chin = landmarks_3d[152]         # 下巴
        left_eye = landmarks_3d[33]      # 左眼外角
        right_eye = landmarks_3d[263]    # 右眼外角

        # 构建面部坐标系
        # x 轴：左眼 → 右眼
        x_axis = right_eye - left_eye
        x_norm = np.linalg.norm(x_axis)
        if x_norm > 1e-8:
            x_axis = x_axis / x_norm
        else:
            x_axis = np.array([1.0, 0.0, 0.0])

        # y 轴大致方向：鼻尖 → 下巴
        y_rough = chin - nose_tip
        # z 轴 = x × y（面部法向量，朝向摄像头）
        z_axis = np.cross(x_axis, y_rough)
        z_norm = np.linalg.norm(z_axis)
        if z_norm > 1e-8:
            z_axis = z_axis / z_norm
        else:
            z_axis = np.array([0.0, 0.0, 1.0])

        # 重新正交化 y 轴
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm > 1e-8:
            y_axis = y_axis / y_norm
        else:
            y_axis = np.array([0.0, 1.0, 0.0])

        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        return rotation_matrix
