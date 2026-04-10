"""
可视化模块 —— 在图像上绘制关键点、指标数值和诊断提示。
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config


class Visualizer:
    """将检测结果绘制到图像帧上。"""

    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[int, int]],
        indices: List[int],
        color: Tuple[int, int, int] = None,
        radius: int = None,
    ) -> None:
        """在指定索引的关键点位置画圆点。"""
        color = color or config.LANDMARK_COLOR
        radius = radius or config.LANDMARK_RADIUS
        for idx in indices:
            if idx < len(landmarks):
                cv2.circle(image, landmarks[idx], radius, color, -1)

    def draw_metrics(
        self,
        image: np.ndarray,
        metrics: Dict[str, Optional[float]],
    ) -> None:
        """在图像左上角显示 PFH 数值和下垂警告。"""
        y_offset = 30

        # 左眼
        left_pfh = metrics.get("left_pfh")
        left_ptosis = metrics.get("left_ptosis", False)
        if left_pfh is not None:
            color = config.TEXT_COLOR_WARNING if left_ptosis else config.TEXT_COLOR_NORMAL
            text = f"Left PFH: {left_pfh:.1f} px"
            if left_ptosis:
                text += "  [Suspected Ptosis]"
            cv2.putText(
                image, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.TEXT_FONT_SCALE, color, config.TEXT_THICKNESS,
            )
            y_offset += 30

        # 右眼
        right_pfh = metrics.get("right_pfh")
        right_ptosis = metrics.get("right_ptosis", False)
        if right_pfh is not None:
            color = config.TEXT_COLOR_WARNING if right_ptosis else config.TEXT_COLOR_NORMAL
            text = f"Right PFH: {right_pfh:.1f} px"
            if right_ptosis:
                text += "  [Suspected Ptosis]"
            cv2.putText(
                image, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.TEXT_FONT_SCALE, color, config.TEXT_THICKNESS,
            )
            y_offset += 30

        # 阈值参考线
        cv2.putText(
            image,
            f"Threshold: {config.PFH_PTOSIS_THRESHOLD} px",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.TEXT_FONT_SCALE * 0.8,
            (180, 180, 180),
            config.TEXT_THICKNESS,
        )

    def draw_no_face(self, image: np.ndarray) -> None:
        """未检测到人脸时的提示。"""
        cv2.putText(
            image, "No face detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.TEXT_FONT_SCALE, (0, 255, 255), config.TEXT_THICKNESS,
        )
