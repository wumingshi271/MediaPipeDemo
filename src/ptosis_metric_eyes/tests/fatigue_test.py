"""疲劳测试流程。"""

import time
from typing import List

import cv2
import numpy as np

from .. import config
from ..detection.mediapipe_detector import MediaPipeFaceDetector
from ..gaze.spherical_eye_tracker import SphericalEyeTracker
from ..metrics.geometric_ptosis import GeometricPtosisCalculator
from ..pose.procrustes_headpose import SimplifiedHeadPoseEstimator
from ..ui.drawing import draw_hud, draw_landmarks, draw_no_face


def run_fatigue_test(
    cap: cv2.VideoCapture,
    detector: MediaPipeFaceDetector,
    head_estimator: SimplifiedHeadPoseEstimator,
    ptosis_calc: GeometricPtosisCalculator,
    eye_tracker: SphericalEyeTracker,
) -> None:
    """30 秒疲劳测试。

    连续记录 PFH，结束后使用 Matplotlib 绘制曲线图。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps: List[float] = []
    pfh_values: List[float] = []

    start_time = time.time()
    duration = config.FATIGUE_TEST_DURATION_SEC
    print(f"[FATIGUE] 疲劳测试开始，持续 {duration} 秒...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        remaining = duration - elapsed
        face_mesh = detector.detect(frame)

        if face_mesh is not None:
            head_pose = head_estimator.estimate(face_mesh)
            ptosis = ptosis_calc.compute(head_pose)
            gaze = eye_tracker.get_gaze_angles(face_mesh, head_pose)

            timestamps.append(elapsed)
            pfh_values.append(ptosis.pfh_mm)

            draw_landmarks(frame, face_mesh)
            draw_hud(
                frame,
                ptosis,
                gaze,
                status_text=f"FATIGUE TEST: {remaining:.1f}s remaining",
            )
        else:
            draw_no_face(frame)
            cv2.putText(
                frame,
                f"FATIGUE TEST: {remaining:.1f}s remaining",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                config.TEXT_COLOR_INFO,
                1,
            )

        cv2.imshow("PtosisMetricEyes", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if len(timestamps) > 1:
        ts = np.array(timestamps)
        pf = np.array(pfh_values)

        coeffs = np.polyfit(ts, pf, 1)
        slope = coeffs[0]
        fit_line = np.polyval(coeffs, ts)

        plt.figure(figsize=(10, 5))
        plt.plot(ts, pf, "b-", alpha=0.7, label="PFH (mm)")
        plt.plot(ts, fit_line, "r--", label=f"Linear fit (slope={slope:.4f} mm/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("PFH (mm)")
        plt.title("Fatigue Test - PFH over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(config.FATIGUE_PLOT_PATH, dpi=150)
        plt.close()

        print(f"[FATIGUE] 测试完成。斜率: {slope:.4f} mm/s")
        print(f"[FATIGUE] 曲线图已保存至 {config.FATIGUE_PLOT_PATH}")

        plot_img = cv2.imread(config.FATIGUE_PLOT_PATH)
        if plot_img is not None:
            cv2.imshow("Fatigue Test Result", plot_img)
            cv2.waitKey(3000)
            cv2.destroyWindow("Fatigue Test Result")
    else:
        print("[FATIGUE] 数据不足，无法绘制曲线")
