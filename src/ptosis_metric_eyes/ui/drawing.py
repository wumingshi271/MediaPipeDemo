"""OpenCV HUD and landmark drawing helpers."""

from typing import Optional

import cv2
import numpy as np

from .. import config
from ..interfaces import FaceMeshResult, GazeResult, PtosisMetrics


def draw_hud(
    frame: np.ndarray,
    ptosis: Optional[PtosisMetrics],
    gaze: Optional[GazeResult],
    status_text: str = "",
) -> None:
    """在画面左上角绘制实时数据 HUD。"""
    y = config.HUD_Y_START
    x = config.HUD_X
    lh = config.HUD_LINE_HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = config.TEXT_FONT_SCALE
    th = config.TEXT_THICKNESS

    if ptosis is not None:
        pfh_color = (
            config.TEXT_COLOR_WARNING
            if ptosis.pfh_mm < config.PFH_THRESHOLD_MM
            else config.TEXT_COLOR_NORMAL
        )
        cv2.putText(
            frame,
            f"PFH: {ptosis.pfh_mm:.1f} mm (L:{ptosis.left_pfh_mm:.1f} R:{ptosis.right_pfh_mm:.1f})",
            (x, y),
            font,
            fs,
            pfh_color,
            th,
        )
        y += lh

        cv2.putText(
            frame,
            f"IPD: {ptosis.ipd_mm:.1f} mm",
            (x, y),
            font,
            fs,
            config.TEXT_COLOR_NORMAL,
            th,
        )
        y += lh

    if gaze is not None:
        cv2.putText(
            frame,
            f"Gaze: yaw {gaze.avg_yaw_deg:.1f} deg, pitch {gaze.avg_pitch_deg:.1f} deg",
            (x, y),
            font,
            fs,
            config.TEXT_COLOR_NORMAL,
            th,
        )
        y += lh

    if ptosis is not None:
        if (
            ptosis.left_pfh_mm < config.PFH_THRESHOLD_MM
            or ptosis.right_pfh_mm < config.PFH_THRESHOLD_MM
        ):
            status = "Suspected Ptosis"
            color = config.TEXT_COLOR_WARNING
        else:
            status = "Normal"
            color = config.TEXT_COLOR_NORMAL
        cv2.putText(frame, f"Status: {status}", (x, y), font, fs, color, th)
        y += lh

    if status_text:
        cv2.putText(frame, status_text, (x, y), font, fs, config.TEXT_COLOR_INFO, th)


def draw_landmarks(frame: np.ndarray, face_mesh: FaceMeshResult) -> None:
    """绘制眼部轮廓和虹膜中心。"""
    px = face_mesh.landmarks_px

    for indices in [config.LEFT_EYE_CONTOUR, config.RIGHT_EYE_CONTOUR]:
        for idx in indices:
            if idx < len(px):
                cv2.circle(
                    frame,
                    tuple(px[idx]),
                    config.LANDMARK_RADIUS,
                    config.LANDMARK_COLOR,
                    -1,
                )

    cv2.circle(
        frame,
        face_mesh.left_iris_center_px,
        config.IRIS_RADIUS,
        config.IRIS_COLOR,
        -1,
    )
    cv2.circle(
        frame,
        face_mesh.right_iris_center_px,
        config.IRIS_RADIUS,
        config.IRIS_COLOR,
        -1,
    )


def draw_no_face(frame: np.ndarray) -> None:
    """未检测到人脸时的提示。"""
    cv2.putText(
        frame,
        "No face detected",
        (config.HUD_X, config.HUD_Y_START),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.TEXT_FONT_SCALE,
        config.TEXT_COLOR_INFO,
        config.TEXT_THICKNESS,
    )

