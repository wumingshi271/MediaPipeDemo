# 任务：在现有 PtosisMetricEyes 项目中增加提上睑肌肌力（Berke法）自动测量功能

## 背景
当前项目已经实现了：
- MediaPipe 人脸关键点检测（FaceDetector）
- 公制头部姿态估计（HeadPoseMetricEstimator，输出 mm 级面部网格）
- 眼球角度解算（EyeTracker）
- 睑裂高度（PFH）和瞳距（IPD）计算（PtosisMetricCalculator）
- 9点个性化校准（Personalizer）
- 疲劳测试模式（按 f 键）

现需要增加 **提上睑肌肌力测量（Berke法）** 功能，操作如下：
1. 固定额肌（模拟医生按压眉弓）：通过检测眉弓关键点位移来验证用户未抬眉。
2. 测量基线：用户向下看，记录上睑缘位置（mm）。
3. 测量高点：用户向上看，记录上睑缘位置（mm）。
4. 计算肌力 = 向上看时的上睑缘垂直坐标 - 向下看时的上睑缘垂直坐标（单位 mm）。
5. 分级：良好 ≥8 mm，中等 4-7 mm，差 ≤4 mm。

## 修改要求（低耦合、不破坏现有结构）

### 1. 新增抽象接口
在 `interfaces.py` 中添加 `MuscleStrengthTester` 抽象基类：

```python
class MuscleStrengthTester(ABC):
    @abstractmethod
    def start_test(self) -> None:
        """初始化测试，显示指导文本，重置内部状态"""
        pass
    
    @abstractmethod
    def capture_down_gaze(self, landmarks_mm: np.ndarray, head_pose: HeadPoseResult) -> None:
        """用户按下空格时调用，记录向下看时的上睑缘3D坐标"""
        pass
    
    @abstractmethod
    def capture_up_gaze(self, landmarks_mm: np.ndarray, head_pose: HeadPoseResult) -> None:
        """用户按下空格时调用，记录向上看时的上睑缘3D坐标"""
        pass
    
    @abstractmethod
    def compute_strength(self) -> float:
        """返回肌力值（mm），如果两次记录不全则返回 None"""
        pass
    
    @abstractmethod
    def get_grade(self, strength_mm: float) -> str:
        """返回分级字符串"""
        pass
```

### 2. 实现 BerkeMuscleTester 类
新建文件 `berke_muscle_tester.py`，实现上述接口。关键点：
- 从 `config.py` 读取上睑缘关键点索引（左眼：159，右眼：386）和眉弓关键点索引（左眉：107，右眉：336）。
- 在 `capture_down_gaze` 和 `capture_up_gaze` 中，根据传入的 `landmarks_mm`（公制面部网格，shape (468, 3)）提取上睑缘的 y 坐标（垂直方向），可左右眼平均或取最小值。
- **额肌阻断验证**：在 `start_test` 时记录眉弓初始 y 坐标。每次 capture 时计算当前眉弓 y 坐标与初始值的差值（绝对值），若超过阈值（如 2 mm），则抛出异常或设置标志位，提示用户“检测到抬眉，请重新测试”。
- 头部稳定验证：可检查 `head_pose` 的平移和旋转变化，若变化过大也提示。
- 测试状态机：用枚举表示 `IDLE`, `WAITING_DOWN`, `WAITING_UP`, `COMPLETED`。

### 3. 修改 config.py
添加以下配置项：
```python
# 肌力测试相关
LEFT_UPPER_LID_MARGIN = 159   # 左眼上睑缘关键点索引
RIGHT_UPPER_LID_MARGIN = 386  # 右眼上睑缘关键点索引
LEFT_BROW_RIDGE = 107         # 左眉弓关键点（用于检测抬眉）
RIGHT_BROW_RIDGE = 336        # 右眉弓关键点
BROW_RISE_THRESHOLD_MM = 2.0  # 眉弓垂直位移阈值（mm）
HEAD_MOVEMENT_THRESHOLD_MM = 10.0  # 头部平移阈值
STRENGTH_GOOD = 8.0
STRENGTH_MODERATE_LOW = 4.0
STRENGTH_MODERATE_HIGH = 7.0
```

### 4. 修改 main.py
- 导入 `BerkeMuscleTester`。
- 在 `__init__` 或主循环前实例化一个 `MuscleStrengthTester` 对象。
- 添加一个测试状态变量 `test_mode`，取值为 `None`, `'muscle'`。
- 在键盘事件中，当按下 `m` 键时：
  - 如果当前不在测试模式，则调用 `tester.start_test()`，设置 `test_mode = 'muscle'`，并显示提示“向下看，然后按空格记录下视位置”。
- 在主循环中，如果 `test_mode == 'muscle'`：
  - 实时显示当前上睑缘位置（mm）以及提示文字。
  - 检测空格键：根据当前状态调用 `tester.capture_down_gaze` 或 `capture_up_gaze`，并更新界面提示。
  - 当两次捕获完成后，调用 `compute_strength` 和 `get_grade`，在屏幕上显示结果（如“肌力: 10.2 mm 分级: 良好”），然后等待任意键退出测试模式（或自动延迟 3 秒后退出）。
- 在测试过程中，需要持续将 `landmarks_mm` 和 `head_pose` 传递给 tester 的捕获方法（捕获时调用）。
- 注意：空格键原本可能用于其他功能（校准中的记录），请确保在肌力测试模式下优先处理肌力捕获，不影响其他模式。

### 5. 辅助函数
- 可能需要从 `geometric_ptosis.py` 中复用获取上睑缘坐标的逻辑，或者直接在 `BerkeMuscleTester` 中实现一个 `_get_upper_lid_y_mm(landmarks_mm)` 方法。

### 6. 更新 README.md
- 添加肌力测试的操作说明：按 `m` 键启动，按照屏幕提示向下看、向上看并按空格，系统自动计算肌力并分级。

## 输出要求
请直接修改现有项目的代码文件，不要重写整个项目。仅输出需要新增或修改的文件内容（可以使用 diff 格式，或者直接给出完整的新文件内容）。重点输出：
- `interfaces.py` 的补充部分
- 完整的 `berke_muscle_tester.py`
- `config.py` 新增配置
- `main.py` 中与肌力测试相关的修改部分（可以用代码块标明在哪里插入）

请保持代码风格与现有项目一致，所有新增代码必须有类型注解和 docstring。
```