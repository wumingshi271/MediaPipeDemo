"""
mediapipe_detector.py — 基于 MediaPipe FaceLandmarker (Tasks API) 的人脸检测实现。

扩展点：
  替换为深度学习分割模型时，实现新的 FaceDetector 子类即可，
  返回相同格式的 FaceMeshResult。
"""

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .. import config
from ..interfaces import FaceDetector, FaceMeshResult


class MediaPipeFaceDetector(FaceDetector):
    """基于 MediaPipe FaceLandmarker 的人脸关键点检测器。

    使用 Tasks API，支持虹膜追踪（478 个关键点）。
    运行模式为 VIDEO，适合逐帧处理视频流。
    """

    def __init__(self) -> None:
        base_options = mp.tasks.BaseOptions(
            model_asset_path=config.FACEMESH_MODEL_PATH
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=config.FACEMESH_MAX_NUM_FACES,
            min_face_detection_confidence=config.FACEMESH_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.FACEMESH_MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.FACEMESH_MIN_TRACKING_CONFIDENCE,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, image: np.ndarray) -> Optional[FaceMeshResult]:
        """检测人脸，返回 478 个 3D 关键点和虹膜中心像素坐标。

        Args:
            image: BGR 格式图像 (H, W, 3)。

        Returns:
            FaceMeshResult 或 None。
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # 递增时间戳（~30fps）
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        num_landmarks = len(face)

        # 提取 3D 归一化坐标 (x, y, z 都在 [0,1] 附近)
        landmarks_3d = np.array(
            [(lm.x, lm.y, lm.z) for lm in face], dtype=np.float64
        )

        # 像素坐标
        landmarks_px = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in face], dtype=np.int32
        )

        # 虹膜中心（如果有 478 个点，索引 468 和 473；否则用眼部中心近似）
        if num_landmarks >= 478:
            left_iris_center = self._to_pixel(face[config.LEFT_IRIS_CENTER], w, h)
            right_iris_center = self._to_pixel(face[config.RIGHT_IRIS_CENTER], w, h)
        else:
            # 回退：使用眼部轮廓中心作为近似
            left_iris_center = self._eye_center_fallback(face, config.LEFT_EYE_CONTOUR, w, h)
            right_iris_center = self._eye_center_fallback(face, config.RIGHT_EYE_CONTOUR, w, h)

        return FaceMeshResult(
            landmarks_3d=landmarks_3d,
            landmarks_px=landmarks_px,
            left_iris_center_px=left_iris_center,
            right_iris_center_px=right_iris_center,
            image_shape=(h, w),
        )

    @staticmethod
    def _to_pixel(landmark, w: int, h: int) -> Tuple[int, int]:
        """将 MediaPipe 归一化坐标转换为像素坐标。"""
        return (int(landmark.x * w), int(landmark.y * h))

    @staticmethod
    def _eye_center_fallback(
        face_landmarks, contour_indices: list, w: int, h: int
    ) -> Tuple[int, int]:
        """当虹膜点不可用时，取眼部轮廓中心作为近似。"""
        xs = [face_landmarks[i].x for i in contour_indices if i < len(face_landmarks)]
        ys = [face_landmarks[i].y for i in contour_indices if i < len(face_landmarks)]
        cx = int(np.mean(xs) * w)
        cy = int(np.mean(ys) * h)
        return (cx, cy)

    def release(self) -> None:
        """释放 MediaPipe 资源。"""
        self._landmarker.close()
