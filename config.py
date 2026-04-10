"""
配置参数模块 —— 集中管理所有可调参数。

扩展点：添加新指标时，在此处增加对应的关键点索引和阈值即可。
"""

# ======================== 输入源 ========================
# 摄像头编号（0 = 默认摄像头），也可改为视频文件路径，如 "video.mp4"
VIDEO_SOURCE = 0

# ======================== MediaPipe 参数 ========================
FACEMESH_MODEL_PATH = "face_landmarker.task"  # 模型文件，相对于项目根目录
FACEMESH_MAX_NUM_FACES = 1
FACEMESH_MIN_DETECTION_CONFIDENCE = 0.5
FACEMESH_MIN_TRACKING_CONFIDENCE = 0.5

# ======================== 眼部关键点索引 (MediaPipe FaceMesh 468+) ========================
# 参考: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
#
# 每只眼选取多对上/下眼睑点，取平均以提高鲁棒性。
# 格式: [(上睑点, 下睑点), ...]

LEFT_EYE_LID_PAIRS = [
    (159, 145),  # 中央
    (158, 153),  # 内侧偏中
    (160, 144),  # 外侧偏中
]

RIGHT_EYE_LID_PAIRS = [
    (386, 374),  # 中央
    (385, 380),  # 内侧偏中
    (387, 373),  # 外侧偏中
]

# 用于绘制的所有眼部关键点（上睑 + 下睑展开）
LEFT_EYE_LANDMARK_INDICES = sorted(
    set(idx for pair in LEFT_EYE_LID_PAIRS for idx in pair)
)
RIGHT_EYE_LANDMARK_INDICES = sorted(
    set(idx for pair in RIGHT_EYE_LID_PAIRS for idx in pair)
)

# ======================== 下垂判断阈值 ========================
PFH_PTOSIS_THRESHOLD = 20  # 像素，低于此值判定为"疑似下垂"

# ======================== 可视化参数 ========================
LANDMARK_COLOR = (0, 255, 0)      # BGR 绿色
LANDMARK_RADIUS = 2
TEXT_COLOR_NORMAL = (255, 255, 255)   # 白色
TEXT_COLOR_WARNING = (0, 0, 255)      # 红色
TEXT_FONT_SCALE = 0.6
TEXT_THICKNESS = 1
