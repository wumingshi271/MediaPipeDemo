"""
config.py — 集中管理所有可调参数。

修改关键点索引、阈值、校准参数等均在此文件中完成，
其他模块通过 import config 引用。
"""

import os

# ======================== 输入源 ========================
# 摄像头编号（0 = 默认摄像头），也可改为视频文件路径，如 "video.mp4"
VIDEO_SOURCE = 0

# ======================== MediaPipe FaceLandmarker 参数 ========================
FACEMESH_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
)
FACEMESH_MAX_NUM_FACES = 1
FACEMESH_MIN_DETECTION_CONFIDENCE = 0.5
FACEMESH_MIN_FACE_PRESENCE_CONFIDENCE = 0.5
FACEMESH_MIN_TRACKING_CONFIDENCE = 0.5

# ======================== 关键点索引 (MediaPipe FaceMesh 478 点) ========================
# 眼睑关键点——用于 PFH 计算
LEFT_EYE_UPPER = 159    # 左眼上睑中央
LEFT_EYE_LOWER = 23     # 左眼下睑中央（need.md 指定）
RIGHT_EYE_UPPER = 386   # 右眼上睑中央
RIGHT_EYE_LOWER = 253   # 右眼下睑中央（need.md 指定）

# 内眼角——用于 IPD 计算
LEFT_INNER_CORNER = 133
RIGHT_INNER_CORNER = 362

# 虹膜关键点索引（MediaPipe 478 点中，468-477 为虹膜点）
# 左眼虹膜: 468(中心), 469-472(边缘)
# 右眼虹膜: 473(中心), 474-477(边缘)
LEFT_IRIS_CENTER = 468
LEFT_IRIS_EDGE_INDICES = [469, 470, 471, 472]
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_EDGE_INDICES = [474, 475, 476, 477]

# 额外眼部关键点（用于绘制）
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# 用于 Procrustes/头部姿态的稳定关键点
# 鼻尖、下巴、左右眼角、左右嘴角
HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# ======================== 生理先验值 ========================
IRIS_DIAMETER_MM = 11.8           # 人类虹膜平均直径（毫米）
DEFAULT_DEPTH_MM = 500.0          # 首帧默认深度假设（毫米）
EYE_RADIUS_MM = (IRIS_DIAMETER_MM / 2) * (12.2 / 5.9)  # ≈ 12.2 mm

# ======================== 下垂判断阈值 ========================
PFH_THRESHOLD_MM = 8.0  # 低于此值判定为"疑似下垂"（毫米）

# ======================== 校准参数 ========================
CALIBRATION_POINTS = 9   # 校准点数量
# 9 个校准点在屏幕上的相对位置（百分比），3x3 网格
CALIBRATION_SCREEN_POINTS = [
    (0.15, 0.15), (0.50, 0.15), (0.85, 0.15),  # 上排
    (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),  # 中排
    (0.15, 0.85), (0.50, 0.85), (0.85, 0.85),  # 下排
]
CALIBRATION_POINT_RADIUS = 20   # 校准红点半径（像素）
CALIBRATION_POINT_COLOR = (0, 0, 255)  # BGR 红色

# ======================== 疲劳测试参数 ========================
FATIGUE_TEST_DURATION_SEC = 30   # 疲劳测试时长（秒）
FATIGUE_PLOT_PATH = "fatigue_plot.png"

# ======================== 肌力测试参数（Berke 法） ========================
LEFT_UPPER_LID_MARGIN = 159    # 左眼上睑缘关键点索引
RIGHT_UPPER_LID_MARGIN = 386   # 右眼上睑缘关键点索引
LEFT_BROW_RIDGE = 107          # 左眉弓关键点（用于检测抬眉）
RIGHT_BROW_RIDGE = 336         # 右眉弓关键点
BROW_RISE_THRESHOLD_MM = 2.0   # 眉弓垂直位移阈值（mm），超过则判定抬眉
HEAD_MOVEMENT_THRESHOLD_MM = 10.0  # 头部平移阈值（mm），超过则提示头动过大
STRENGTH_GOOD = 8.0            # 肌力分级：良好 ≥ 8 mm
STRENGTH_MODERATE_LOW = 4.0    # 肌力分级：中等 4-7 mm
STRENGTH_MODERATE_HIGH = 7.0   # 肌力分级：中等上限

# ======================== 可视化参数 ========================
LANDMARK_COLOR = (0, 255, 0)         # BGR 绿色
LANDMARK_RADIUS = 1
IRIS_COLOR = (255, 255, 0)           # BGR 青色
IRIS_RADIUS = 2
TEXT_COLOR_NORMAL = (255, 255, 255)  # 白色
TEXT_COLOR_WARNING = (0, 0, 255)     # 红色
TEXT_COLOR_INFO = (0, 255, 255)      # 黄色
TEXT_FONT_SCALE = 0.55
TEXT_THICKNESS = 1
HUD_LINE_HEIGHT = 25                 # HUD 行高（像素）
HUD_X = 10                          # HUD 文字 x 起始位置
HUD_Y_START = 25                    # HUD 文字 y 起始位置
