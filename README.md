# PtosisMetricEyes

融合上睑下垂检测与视线追踪的基础版 Demo。

## 功能

- **实时显示**：睑裂高度 PFH (mm)、瞳距 IPD (mm)、眼球偏航角/俯仰角 (deg)
- **9 点校准**：按 `c` 键启动，优化眼球半径等个人参数
- **疲劳测试**：按 `f` 键启动，录制 30 秒 PFH 曲线并生成图表
- **下垂提示**：PFH < 8mm 时显示红色警告
- **肌力测试**：按 `m` 键启动 Berke 法提上睑肌肌力测量，自动分级

## 安装

```bash
pip install -r requirements.txt
```

需要模型文件 `face_landmarker.task`（已包含在项目中）。如需重新下载：

```bash
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## 运行

```bash
# 使用默认摄像头
python main.py

# 使用视频文件
python main.py video.mp4
```

## 操作

| 按键 | 功能 |
|------|------|
| `c`  | 启动 9 点校准 |
| `f`  | 启动 30 秒疲劳测试 |
| `m`  | 启动提上睑肌肌力测试（Berke 法） |
| `q`  | 退出 |

### 肌力测试操作流程

1. 按 `m` 键进入肌力测试模式
2. **向下看**，保持头部不动，按**空格键**记录下视位置
3. **向上看**，保持头部不动、不抬眉，按**空格键**记录上视位置
4. 系统自动计算肌力（mm）并显示分级：良好(≥8mm) / 中等(4-7mm) / 差(≤4mm)

> 注意：测试过程中系统会监测眉弓位移和头部移动。若检测到抬眉或头动过大会提示重试。

## 项目结构

| 文件 | 说明 |
|------|------|
| `interfaces.py` | 抽象基类与数据类定义 |
| `config.py` | 所有可调参数 |
| `mediapipe_detector.py` | MediaPipe FaceLandmarker 实现 |
| `procrustes_headpose.py` | 公制头部姿态估计（简化版） |
| `spherical_eye_tracker.py` | 球形眼球模型视线追踪 |
| `geometric_ptosis.py` | PFH/IPD 几何计算 |
| `simple_personalizer.py` | 9 点校准优化 |
| `berke_muscle_tester.py` | 提上睑肌肌力测试（Berke 法） |
| `main.py` | 主程序入口 |
