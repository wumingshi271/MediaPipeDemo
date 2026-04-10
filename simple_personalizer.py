"""
simple_personalizer.py — 基于 9 点校准的个性化参数优化。

校准流程：
  1. 用户注视屏幕上 9 个已知位置的红点，按空格键采集数据。
  2. 收集每个采集点的 head_pose、iris_center_px、pfh_pixel、screen_point。
  3. 使用最小二乘优化眼球半径缩放因子和上眼睑垂直偏移。

优化目标：
  对于每个校准点，使得基于球形模型预测的视线方向尽可能指向屏幕上的已知点。
  简化版直接使用线性回归拟合眼球半径。
"""

from typing import Dict, List

import numpy as np

import config
from interfaces import Personalizer


class SimplePersonalizer(Personalizer):
    """基于最小二乘的简单个性化校准器。

    优化以下参数：
      - eye_radius_scale: 眼球半径缩放因子（初始 1.0）
      - lid_offset_mm: 上眼睑垂直偏移（毫米）
    """

    def calibrate(self, calibration_data: List[Dict]) -> Dict:
        """使用校准数据优化个人参数。

        Args:
            calibration_data: 列表，每个元素包含:
                - "screen_point": (sx, sy) 屏幕归一化坐标
                - "iris_center_left_px": (x, y) 左眼虹膜中心像素
                - "iris_center_right_px": (x, y) 右眼虹膜中心像素
                - "head_pose": HeadPoseResult
                - "pfh_mm": 当前帧 PFH（毫米）

        Returns:
            优化后的参数字典:
                {"eye_radius_scale": float, "lid_offset_mm": float}
        """
        if len(calibration_data) < 3:
            print("[WARNING] 校准数据不足 3 个点，使用默认参数")
            return {"eye_radius_scale": 1.0, "lid_offset_mm": 0.0}

        try:
            return self._optimize_with_scipy(calibration_data)
        except ImportError:
            print("[WARNING] scipy 未安装，使用简化校准")
            return self._simple_calibrate(calibration_data)

    def _optimize_with_scipy(self, calibration_data: List[Dict]) -> Dict:
        """使用 scipy.optimize.least_squares 优化参数。"""
        from scipy.optimize import least_squares

        # 提取屏幕点和虹膜像素位移
        screen_xs = np.array([d["screen_point"][0] for d in calibration_data])
        screen_ys = np.array([d["screen_point"][1] for d in calibration_data])

        # 使用左眼虹膜位移作为优化目标
        iris_xs = np.array([d["iris_center_left_px"][0] for d in calibration_data])
        iris_ys = np.array([d["iris_center_left_px"][1] for d in calibration_data])

        # 中心化
        iris_xs_c = iris_xs - np.mean(iris_xs)
        iris_ys_c = iris_ys - np.mean(iris_ys)
        screen_xs_c = screen_xs - np.mean(screen_xs)
        screen_ys_c = screen_ys - np.mean(screen_ys)

        def residuals(params):
            scale = params[0]
            # 预测的屏幕坐标 ∝ iris_displacement * scale
            pred_x = iris_xs_c * scale
            pred_y = iris_ys_c * scale
            res_x = pred_x - screen_xs_c
            res_y = pred_y - screen_ys_c
            return np.concatenate([res_x, res_y])

        result = least_squares(residuals, x0=[0.001], bounds=([1e-6], [1.0]))
        optimal_scale = result.x[0]

        # 将尺度因子转换为眼球半径缩放
        # 较大的 optimal_scale → 虹膜位移映射到更大的角度 → 较小的眼球半径
        eye_radius_scale = 1.0 / (optimal_scale * 1000.0 + 1.0)
        eye_radius_scale = np.clip(eye_radius_scale, 0.5, 2.0)

        # PFH 偏移（取校准数据平均 PFH 与默认阈值的差异）
        pfh_values = [d.get("pfh_mm", config.PFH_THRESHOLD_MM) for d in calibration_data]
        mean_pfh = np.mean(pfh_values)
        lid_offset_mm = 0.0  # 简化版不修正偏移

        print(f"[CALIBRATION] 眼球半径缩放: {eye_radius_scale:.3f}, "
              f"平均 PFH: {mean_pfh:.2f} mm")

        return {
            "eye_radius_scale": float(eye_radius_scale),
            "lid_offset_mm": float(lid_offset_mm),
        }

    @staticmethod
    def _simple_calibrate(calibration_data: List[Dict]) -> Dict:
        """简化校准：不依赖 scipy，直接统计。"""
        pfh_values = [d.get("pfh_mm", config.PFH_THRESHOLD_MM) for d in calibration_data]
        mean_pfh = np.mean(pfh_values)

        print(f"[CALIBRATION] 简化模式，平均 PFH: {mean_pfh:.2f} mm")
        return {
            "eye_radius_scale": 1.0,
            "lid_offset_mm": 0.0,
        }
