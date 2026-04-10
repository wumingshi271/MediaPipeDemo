```text
# 任务目标
请用 Python 实现一个上睑下垂（ptosis）检测的基础版 demo，使用 MediaPipe 进行人脸关键点检测，通过几何计算获得眼睑裂高度（PFH）和简单的下垂判断。要求代码耦合度低，模块化设计，方便后续替换或升级为深度学习分割模型（例如 PeriorbitAI 或 AutoPtosis）。

## 技术栈
- Python 3.8+
- OpenCV (图像/视频处理)
- MediaPipe (人脸关键点检测)
- NumPy (坐标计算)

## 功能需求
1. 从摄像头或视频文件中读取图像帧。
2. 对每一帧进行人脸检测，获取 MediaPipe FaceMesh 的 468 个关键点。
3. 根据预定义的眼部关键点索引（左眼和右眼的上、下眼睑关键点）计算睑裂高度（像素距离）。
4. 在图像上绘制关键点、显示睑裂高度值，并基于简单阈值（例如 PFH < 20 像素）输出“疑似下垂”的文本提示。
5. 按 'q' 键退出程序。

## 低耦合与可拓展性要求
- 请将核心功能拆分为独立的模块或类，至少包括：
  - `FaceLandmarkDetector`：封装 MediaPipe 的初始化、推理、关键点获取。
  - `PtosisMetricsCalculator`：输入关键点坐标，返回 PFH、MRD-1（可选）等测量值。当前只实现 PFH，但设计成可轻松添加新指标。
  - `Visualizer`：负责在图像上绘制关键点、数值和诊断文字。
  - `main.py`：仅负责读取视频流、协调上述模块、控制主循环。
- 定义清晰的接口（例如使用抽象基类或普通类 + 方法），使得未来替换 FaceLandmarkDetector 为深度学习分割模型时，主流程几乎不需改动。
- 所有配置参数（如关键点索引、PFH 阈值、摄像头编号）应集中在一个配置文件或字典中，而不是硬编码在函数内部。
- 提供简单的 README 说明如何运行（安装依赖、执行命令）。

## 输出要求
请输出完整的代码文件，按以下结构组织（可以直接复制到项目中使用）：
```
project/
│
├── config.py            # 配置参数（关键点索引、阈值、摄像头ID等）
├── detector.py          # FaceLandmarkDetector 实现
├── metrics.py           # PtosisMetricsCalculator 实现
├── visualizer.py        # Visualizer 实现
├── main.py              # 主入口程序
└── requirements.txt     # 依赖列表
```

请确保代码可直接运行，并在注释中说明预留的拓展点（例如如何替换为深度学习模型）。不要使用外部未列出的额外库（标准库除外）。
```