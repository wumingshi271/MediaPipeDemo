"""
上睑下垂（Ptosis）检测 Demo —— 主入口。

使用方式:
    python main.py              # 使用默认摄像头
    python main.py video.mp4    # 使用视频文件

按 'q' 键退出。
"""

import sys

import cv2

import config
from detector import FaceLandmarkDetector
from metrics import PtosisMetricsCalculator
from visualizer import Visualizer


def main() -> None:
    # 支持命令行传入视频文件路径
    source = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_SOURCE

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{source}'")
        sys.exit(1)

    detector = FaceLandmarkDetector()
    calculator = PtosisMetricsCalculator()
    visualizer = Visualizer()

    print("Ptosis Detection Demo started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detector.detect(frame)

        if landmarks is not None:
            # 计算指标
            metrics = calculator.evaluate(landmarks)

            # 绘制眼部关键点
            visualizer.draw_landmarks(
                frame, landmarks, config.LEFT_EYE_LANDMARK_INDICES
            )
            visualizer.draw_landmarks(
                frame, landmarks, config.RIGHT_EYE_LANDMARK_INDICES
            )

            # 绘制指标数值和诊断提示
            visualizer.draw_metrics(frame, metrics)
        else:
            visualizer.draw_no_face(frame)

        cv2.imshow("Ptosis Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
