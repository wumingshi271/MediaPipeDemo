"""
geometric_ptosis.py — 基于公制面部网格的 PFH 和 IPD 几何计算。

从 HeadPoseResult 中的公制面部网格 (mm) 提取上下眼睑、内眼角关键点，
计算 3D 欧氏距离作为睑裂高度 (PFH) 和瞳距 (IPD)。
"""

import numpy as np

from .. import config
from ..interfaces import HeadPoseResult, PtosisMetricCalculator, PtosisMetrics


class GeometricPtosisCalculator(PtosisMetricCalculator):
    """基于几何距离的下垂指标计算器。

    使用公制面部网格（毫米）中的关键点 3D 欧氏距离
    来估算睑裂高度和瞳距。
    """

    def compute(self, head_pose: HeadPoseResult) -> PtosisMetrics:
        """计算 PFH (mm) 和 IPD (mm)。

        Args:
            head_pose: 包含公制面部网格 face_mesh_mm 的姿态结果。

        Returns:
            PtosisMetrics，包含左/右眼 PFH、平均 PFH 和 IPD。
        """
        mesh_mm = head_pose.face_mesh_mm

        # 左眼 PFH: 上睑点 159 → 下睑点 23 的 3D 距离
        left_pfh = self._distance_3d(
            mesh_mm, config.LEFT_EYE_UPPER, config.LEFT_EYE_LOWER
        )

        # 右眼 PFH: 上睑点 386 → 下睑点 253 的 3D 距离
        right_pfh = self._distance_3d(
            mesh_mm, config.RIGHT_EYE_UPPER, config.RIGHT_EYE_LOWER
        )

        # 平均 PFH
        pfh_mm = (left_pfh + right_pfh) / 2.0

        # IPD: 左内眼角 133 → 右内眼角 362 的 3D 距离
        ipd_mm = self._distance_3d(
            mesh_mm, config.LEFT_INNER_CORNER, config.RIGHT_INNER_CORNER
        )

        return PtosisMetrics(
            pfh_mm=round(pfh_mm, 2),
            ipd_mm=round(ipd_mm, 2),
            left_pfh_mm=round(left_pfh, 2),
            right_pfh_mm=round(right_pfh, 2),
        )

    @staticmethod
    def _distance_3d(mesh: np.ndarray, idx_a: int, idx_b: int) -> float:
        """计算网格中两个关键点的 3D 欧氏距离。

        Args:
            mesh: 公制面部网格 (N, 3)，单位毫米。
            idx_a: 关键点 A 索引。
            idx_b: 关键点 B 索引。

        Returns:
            两点间的 3D 距离（毫米）。
        """
        if idx_a >= len(mesh) or idx_b >= len(mesh):
            return 0.0
        return float(np.linalg.norm(mesh[idx_a] - mesh[idx_b]))
