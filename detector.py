"""
人脸关键点检测模块。

扩展点:
  要替换为深度学习分割模型（如 PeriorbitAI），只需继承 BaseLandmarkDetector
  并实现 detect() 方法，返回相同格式的关键点列表，主流程无需改动。
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config


class BaseLandmarkDetector(ABC):
    """关键点检测器的抽象基类。"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        检测人脸关键点。

        Args:
            image: BGR 格式的图像 (H, W, 3)。

        Returns:
            关键点像素坐标列表 [(x, y), ...]，索引与 MediaPipe FaceMesh 对齐。
            未检测到人脸时返回 None。
        """
        ...

    def release(self) -> None:
        """释放资源（可选覆写）。"""
        pass


class FaceLandmarkDetector(BaseLandmarkDetector):
    """基于 MediaPipe FaceLandmarker (Tasks API) 的关键点检测器。"""

    def __init__(self) -> None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config.FACEMESH_MODEL_PATH,
        )
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=config.FACEMESH_MAX_NUM_FACES,
            min_face_detection_confidence=config.FACEMESH_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.FACEMESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACEMESH_MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # ~30fps
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        landmarks = [
            (int(lm.x * w), int(lm.y * h)) for lm in face
        ]
        return landmarks

    def release(self) -> None:
        self._landmarker.close()
