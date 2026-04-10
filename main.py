"""
main.py — PtosisMetricEyes 主程序入口。

功能：
  - 实时视频流处理，显示 PFH(mm)、IPD(mm)、Gaze yaw/pitch
  - 按 'c' 启动 9 点校准
  - 按 'f' 启动 30 秒疲劳测试
  - 按 'm' 启动提上睑肌肌力测试（Berke 法）
  - 按 'q' 退出

使用方式：
    python main.py              # 使用默认摄像头
    python main.py video.mp4    # 使用视频文件
"""

import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from interfaces import FaceMeshResult, GazeResult, HeadPoseResult, PtosisMetrics
from mediapipe_detector import MediaPipeFaceDetector
from procrustes_headpose import SimplifiedHeadPoseEstimator
from spherical_eye_tracker import SphericalEyeTracker
from geometric_ptosis import GeometricPtosisCalculator
from simple_personalizer import SimplePersonalizer
from berke_muscle_tester import BerkeMuscleTester, TestState


# ======================== HUD 绘制 ========================

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
        # PFH
        pfh_color = config.TEXT_COLOR_WARNING if ptosis.pfh_mm < config.PFH_THRESHOLD_MM else config.TEXT_COLOR_NORMAL
        cv2.putText(frame, f"PFH: {ptosis.pfh_mm:.1f} mm (L:{ptosis.left_pfh_mm:.1f} R:{ptosis.right_pfh_mm:.1f})",
                    (x, y), font, fs, pfh_color, th)
        y += lh

        # IPD
        cv2.putText(frame, f"IPD: {ptosis.ipd_mm:.1f} mm",
                    (x, y), font, fs, config.TEXT_COLOR_NORMAL, th)
        y += lh

    if gaze is not None:
        cv2.putText(frame, f"Gaze: yaw {gaze.avg_yaw_deg:.1f} deg, pitch {gaze.avg_pitch_deg:.1f} deg",
                    (x, y), font, fs, config.TEXT_COLOR_NORMAL, th)
        y += lh

    # 状态
    if ptosis is not None:
        if ptosis.left_pfh_mm < config.PFH_THRESHOLD_MM or ptosis.right_pfh_mm < config.PFH_THRESHOLD_MM:
            status = "Suspected Ptosis"
            color = config.TEXT_COLOR_WARNING
        else:
            status = "Normal"
            color = config.TEXT_COLOR_NORMAL
        cv2.putText(frame, f"Status: {status}", (x, y), font, fs, color, th)
        y += lh

    if status_text:
        cv2.putText(frame, status_text, (x, y), font, fs, config.TEXT_COLOR_INFO, th)
        y += lh


def draw_landmarks(
    frame: np.ndarray,
    face_mesh: FaceMeshResult,
) -> None:
    """绘制眼部轮廓和虹膜中心。"""
    px = face_mesh.landmarks_px

    # 眼部轮廓
    for indices in [config.LEFT_EYE_CONTOUR, config.RIGHT_EYE_CONTOUR]:
        for idx in indices:
            if idx < len(px):
                cv2.circle(frame, tuple(px[idx]), config.LANDMARK_RADIUS, config.LANDMARK_COLOR, -1)

    # 虹膜中心
    cv2.circle(frame, face_mesh.left_iris_center_px, config.IRIS_RADIUS, config.IRIS_COLOR, -1)
    cv2.circle(frame, face_mesh.right_iris_center_px, config.IRIS_RADIUS, config.IRIS_COLOR, -1)


def draw_no_face(frame: np.ndarray) -> None:
    """未检测到人脸时的提示。"""
    cv2.putText(frame, "No face detected", (config.HUD_X, config.HUD_Y_START),
                cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_FONT_SCALE,
                config.TEXT_COLOR_INFO, config.TEXT_THICKNESS)


# ======================== 校准流程 ========================

def run_calibration(
    cap: cv2.VideoCapture,
    detector: MediaPipeFaceDetector,
    head_estimator: SimplifiedHeadPoseEstimator,
    ptosis_calc: GeometricPtosisCalculator,
    eye_tracker: SphericalEyeTracker,
    personalizer: SimplePersonalizer,
) -> dict:
    """9 点校准流程。

    显示红色校准点，用户注视后按空格键采集数据。
    采集完 9 个点后自动优化参数。

    Returns:
        优化后的参数字典。
    """
    calibration_data: List[dict] = []
    point_idx = 0
    total_points = len(config.CALIBRATION_SCREEN_POINTS)

    print(f"[CALIBRATION] 开始 {total_points} 点校准，请注视红色圆点后按空格键")

    while point_idx < total_points:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        sx, sy = config.CALIBRATION_SCREEN_POINTS[point_idx]
        target_x = int(sx * w)
        target_y = int(sy * h)

        # 绘制校准点
        cv2.circle(frame, (target_x, target_y),
                   config.CALIBRATION_POINT_RADIUS, config.CALIBRATION_POINT_COLOR, -1)
        cv2.putText(frame, f"Calibration {point_idx + 1}/{total_points}: Look at the red dot, press SPACE",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TEXT_COLOR_INFO, 1)

        # 检测
        face_mesh = detector.detect(frame)
        if face_mesh is not None:
            draw_landmarks(frame, face_mesh)

        cv2.imshow("PtosisMetricEyes", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and face_mesh is not None:
            # 采集当前帧数据
            head_pose = head_estimator.estimate(face_mesh)
            ptosis = ptosis_calc.compute(head_pose)

            # 第一个点时设置正面参考
            if point_idx == 0:
                eye_tracker.set_reference(face_mesh, head_pose)

            calibration_data.append({
                "screen_point": (sx, sy),
                "iris_center_left_px": face_mesh.left_iris_center_px,
                "iris_center_right_px": face_mesh.right_iris_center_px,
                "head_pose": head_pose,
                "pfh_mm": ptosis.pfh_mm,
            })
            print(f"  Point {point_idx + 1} captured.")
            point_idx += 1

        elif key == ord("q"):
            print("[CALIBRATION] 校准被取消")
            return {"eye_radius_scale": 1.0, "lid_offset_mm": 0.0}

    # 执行优化
    params = personalizer.calibrate(calibration_data)
    print(f"[CALIBRATION] 校准完成: {params}")
    return params


# ======================== 疲劳测试 ========================

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
    matplotlib.use("Agg")  # 非交互后端，避免阻塞
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
            draw_hud(frame, ptosis, gaze,
                     status_text=f"FATIGUE TEST: {remaining:.1f}s remaining")
        else:
            draw_no_face(frame)
            cv2.putText(frame, f"FATIGUE TEST: {remaining:.1f}s remaining",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        config.TEXT_COLOR_INFO, 1)

        cv2.imshow("PtosisMetricEyes", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 绘制曲线图
    if len(timestamps) > 1:
        ts = np.array(timestamps)
        pf = np.array(pfh_values)

        # 线性拟合
        coeffs = np.polyfit(ts, pf, 1)
        slope = coeffs[0]
        fit_line = np.polyval(coeffs, ts)

        plt.figure(figsize=(10, 5))
        plt.plot(ts, pf, "b-", alpha=0.7, label="PFH (mm)")
        plt.plot(ts, fit_line, "r--", label=f"Linear fit (slope={slope:.4f} mm/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("PFH (mm)")
        plt.title("Fatigue Test — PFH over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(config.FATIGUE_PLOT_PATH, dpi=150)
        plt.close()

        print(f"[FATIGUE] 测试完成。斜率: {slope:.4f} mm/s")
        print(f"[FATIGUE] 曲线图已保存至 {config.FATIGUE_PLOT_PATH}")

        # 短暂显示结果图
        plot_img = cv2.imread(config.FATIGUE_PLOT_PATH)
        if plot_img is not None:
            cv2.imshow("Fatigue Test Result", plot_img)
            cv2.waitKey(3000)
            cv2.destroyWindow("Fatigue Test Result")
    else:
        print("[FATIGUE] 数据不足，无法绘制曲线")


# ======================== 肌力测试流程 ========================

def run_muscle_test(
    cap: cv2.VideoCapture,
    detector: MediaPipeFaceDetector,
    head_estimator: SimplifiedHeadPoseEstimator,
    ptosis_calc: GeometricPtosisCalculator,
    eye_tracker: SphericalEyeTracker,
    tester: BerkeMuscleTester,
) -> None:
    """提上睑肌肌力测试（Berke 法）流程。

    用户按照屏幕提示向下看/向上看并按空格采集，
    系统自动计算肌力并显示分级结果。
    """
    tester.start_test()

    while tester.state not in (TestState.IDLE, TestState.COMPLETED):
        ret, frame = cap.read()
        if not ret:
            break

        face_mesh = detector.detect(frame)
        head_pose = None
        ptosis = None

        if face_mesh is not None:
            head_pose = head_estimator.estimate(face_mesh)
            ptosis = ptosis_calc.compute(head_pose)
            gaze = eye_tracker.get_gaze_angles(face_mesh, head_pose)

            draw_landmarks(frame, face_mesh)

            # 实时显示上睑缘 y 坐标供参考
            lid_y = tester._get_upper_lid_y_mm(head_pose.face_mesh_mm)
            draw_hud(frame, ptosis, gaze,
                     status_text=f"{tester.prompt_text}  |  Lid Y: {lid_y:.1f} mm")
        else:
            draw_no_face(frame)
            cv2.putText(frame, tester.prompt_text,
                        (config.HUD_X, config.HUD_Y_START + config.HUD_LINE_HEIGHT),
                        cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_FONT_SCALE,
                        config.TEXT_COLOR_INFO, config.TEXT_THICKNESS)

        # 显示警告
        if tester.warning:
            h_frame = frame.shape[0]
            cv2.putText(frame, tester.warning,
                        (config.HUD_X, h_frame - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        config.TEXT_COLOR_WARNING, config.TEXT_THICKNESS)

        cv2.imshow("PtosisMetricEyes", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            tester.state = TestState.IDLE
            print("[MUSCLE] 测试被取消")
            return
        elif key == ord(" ") and face_mesh is not None and head_pose is not None:
            try:
                if tester.state == TestState.WAITING_DOWN:
                    tester.capture_down_gaze(head_pose.face_mesh_mm, head_pose)
                elif tester.state == TestState.WAITING_UP:
                    tester.capture_up_gaze(head_pose.face_mesh_mm, head_pose)
            except ValueError as e:
                tester.warning = str(e)
                print(f"[MUSCLE] WARNING: {e}")

    # 显示结果
    strength = tester.compute_strength()
    if strength is not None:
        grade = tester.get_grade(strength)
        result_text = f"Muscle Strength: {strength:.1f} mm  Grade: {grade}"
        print(f"[MUSCLE] {result_text}")

        # 显示结果 3 秒
        for _ in range(90):  # ~3s at 30fps
            ret, frame = cap.read()
            if not ret:
                break
            face_mesh = detector.detect(frame)
            if face_mesh is not None:
                draw_landmarks(frame, face_mesh)

            grade_color = {
                "Good": (0, 255, 0),
                "Moderate": (0, 200, 255),
                "Poor": config.TEXT_COLOR_WARNING,
            }.get(grade, config.TEXT_COLOR_NORMAL)

            cv2.putText(frame, result_text,
                        (config.HUD_X, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, grade_color, 2)
            cv2.putText(frame, "Press any key to continue...",
                        (config.HUD_X, frame.shape[0] // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("PtosisMetricEyes", frame)
            if cv2.waitKey(33) & 0xFF != 255:
                break
    else:
        print("[MUSCLE] 数据不完整，无法计算肌力")


# ======================== 主循环 ========================

def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_SOURCE

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{source}'")
        sys.exit(1)

    # 初始化模块
    detector = MediaPipeFaceDetector()
    head_estimator = SimplifiedHeadPoseEstimator()
    eye_tracker = SphericalEyeTracker()
    ptosis_calc = GeometricPtosisCalculator()
    personalizer = SimplePersonalizer()
    muscle_tester = BerkeMuscleTester()

    print("PtosisMetricEyes started.")
    print("  'c' = 9-point calibration")
    print("  'f' = 30s fatigue test")
    print("  'm' = muscle strength test (Berke)")
    print("  'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_mesh = detector.detect(frame)

        ptosis: Optional[PtosisMetrics] = None
        gaze: Optional[GazeResult] = None

        if face_mesh is not None:
            head_pose = head_estimator.estimate(face_mesh)
            ptosis = ptosis_calc.compute(head_pose)
            gaze = eye_tracker.get_gaze_angles(face_mesh, head_pose)

            draw_landmarks(frame, face_mesh)
            draw_hud(frame, ptosis, gaze)
        else:
            draw_no_face(frame)

        # 底部操作提示
        h_frame = frame.shape[0]
        cv2.putText(frame, "[c] Calibrate  [f] Fatigue  [m] Muscle Test  [q] Quit",
                    (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1)

        cv2.imshow("PtosisMetricEyes", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            params = run_calibration(
                cap, detector, head_estimator, ptosis_calc, eye_tracker, personalizer
            )
            # 应用校准参数
            eye_tracker.eye_radius_mm = config.EYE_RADIUS_MM * params.get("eye_radius_scale", 1.0)
            print(f"  Updated eye radius: {eye_tracker.eye_radius_mm:.2f} mm")
        elif key == ord("f"):
            run_fatigue_test(cap, detector, head_estimator, ptosis_calc, eye_tracker)
        elif key == ord("m"):
            run_muscle_test(
                cap, detector, head_estimator, ptosis_calc, eye_tracker, muscle_tester
            )

    detector.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
