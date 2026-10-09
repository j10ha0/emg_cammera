# 实时手部关节角度标注程序

固定机位摄像头 → **MediaPipe** 检测 21 个手部关键点 → **实时计算并标注关节角度**。

![示意](out/img_nocontact_Apple_lc.jpg)

---

## 一、快速开始

```bat
:: 一键运行（默认：摄像头0 + 中文面板 + 自动 ROI）
run.bat

:: 或者手动
python main.py --panel --auto-roi
```

**首次运行会自动下载模型** `hand_landmarker.task`（约 7.8 MB）到 `models/`。

### 按键
| 键 | 功能 |
|----|------|
| **q / ESC** | 退出 |
| **s** | 截图到 `out/` |
| **m** | 切换镜像（水平翻转） |
| **空格** | 暂停 / 继续 |
| **r** | 开始 / 停止录制视频 |

---

## 二、界面说明

| 元素 | 含义 |
|------|------|
| 彩色骨架 | 每根手指一种颜色（拇指红 / 食指绿 / 中指黄 / 无名粉 / 小指青） |
| 关节旁的数字 | 该关节角度（度）。**M**=MCP　**P**=PIP　**D**=DIP |
| 左上角文字 | 手别 + 手势提示 + 四指平均弯曲度 |
| 左下角条形 | 每根手指的弯曲度（0=伸直，1=握紧） |
| 右侧面板 | 完整数值表：各指 MCP/PIP/DIP、张开度、拇指-食指对合、掌长掌宽 |

---

## 三、计算的是什么

```
21 个关键点索引
  0        腕
  1-4      拇指: cmc, mcp, ip,  tip
  5-8      食指: mcp, pip, dip, tip
  9-12     中指
  13-16    无名指
  17-20    小指

关节角定义（三点的夹角）
  拇指:   MCP = ∠(cmc, mcp, ip)      IP = ∠(mcp, ip, tip)
  四指:   MCP = ∠(腕, mcp, pip)      PIP = ∠(mcp, pip, dip)     DIP = ∠(pip, dip, tip)

其他
  弯曲度 curl  = 由 PIP 角映射：180°→0.00（伸直）, 60°→1.00（握紧）
  手指张开度   = 相邻指根相对腕的方向夹角
  拇指-食指对合 = |拇指尖 - 食指尖| / 掌长
```

角度用 **`hand_world_landmarks`（手部度量 3D 坐标）** 计算，**与相机距离/角度无关** ——
所以固定机位下旋转相机、挪远挪近，角度值都不变。

---

## 四、命令行参数

```bat
python main.py [选项]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--camera N` | 0 | 摄像头索引（换个 USB 口/摄像头可能要改） |
| `--width` / `--height` | 1280 / 720 | 采集分辨率 |
| `--fps` | 30 | 请求帧率 |
| `--no-mirror` | — | 不水平翻转画面 |
| `--panel` | 关 | 显示右侧中文数据面板 |
| `--no-bars` | — | 不显示弯曲度条形 |
| `--num-hands` | 2 | 最多检测几只手 |
| `--det-conf` | 0.30 | 检测置信度阈值 |
| `--roi x1,y1,x2,y2` | 无 | **限定检测区域**（归一化 0~1），手小的时候用 |
| `--auto-roi` | 关 | 全幅检不到手时，自动尝试若干 ROI |
| `--record` | 关 | 开窗模式下同时录制标注视频 |
| `--log` | 关 | 把角度逐帧写入 `out/angles.csv` |
| `--image PATH` | — | **离线模式**：处理单张图或整个目录（不开摄像头） |
| `--headless N` | 0 | 无窗口：处理 N 帧后退出（自动化测试用） |

### 例子
```bat
run.bat --camera 1                          :: 换摄像头
run.bat --log                               :: 顺便记录角度 CSV
python main.py --image test_images2 --panel :: 离线跑一批图
python main.py --roi 0.3,0.2,0.8,0.9        :: 手只出现在某区域时
python main.py --det-conf 0.2               :: 手很难检到时放宽阈值
```

---

## 五、文件结构

```
cammera/
├── main.py                  主程序：采集 → 检测 → 计算 → 绘制 → 显示
│                            （支持实时 / 离线 --image / 无窗口 --headless）
├── camera.py                ⭐ 摄像头封装（**换 RealSense/Orbbec 只改这个文件**）
├── hand_pose.py             MediaPipe HandLandmarker 封装（Tasks API，自动下载模型）
├── joint_angle.py           关节角度计算（与相机无关）
├── visualize.py             绘制（骨架 / 角度 / 中文面板 / 弯曲度条形）
│
├── capture_poses.py         ⭐ 交互式采集自己的手型（摆姿势按 SPACE 抓拍）
├── make_ppt_pack.py         ⭐ 一键生成 PPT 图包（3 张图：系统/手型/可靠性）
├── make_ppt_figures.py      单张图自动裁"手部特写"并标注
├── make_test_crops.py       从宽幅图里批量裁"手部特写"做测试
├── test_angle.py            用合成关键点验证角度公式是否正确
├── test_angle_robustness.py ⭐ 关节角抗噪分析（图3 的数据来源）
│
├── run.bat                  一键启动（内置 Python 自动搜索）
├── requirements.txt
├── README.md
├── DATA_SOURCES.md          数据来源与许可声明
├── .gitignore
│
├── ppt_figures/             ⭐ PPT 图包（fig1/fig2/fig3 + 说明）
├── test_images/  test_images2/   测试图（原始宽幅 / 手部特写）
├── nc_src/  ppt_src/             测试图源（无遮挡帧 / 各物体抓握帧）
├── models/                  自动下载的 hand_landmarker.task（不入库）
└── out/                     运行时输出：截图 / 标注图 / 角度 CSV / 录制视频（不入库）
```

### 常用命令速查

```bat
run.bat                                    :: 实时显示（推荐）
run.bat --log                              :: 顺便记录角度 CSV
run.bat --record                           :: 顺便录制标注视频
python capture_poses.py                    :: 采集自己的手型（做 PPT 用）
python make_ppt_pack.py                    :: 重新生成 PPT 图包
python test_angle.py                       :: 验证角度公式（应输出"✅ 角度公式正确"）
python test_angle_robustness.py            :: 关角度可靠性分析
python main.py --image 某目录 --panel      :: 离线批量跑图片
```

**分层设计**：`camera.py` 是唯一与相机相关的模块。
以后换深度相机（RealSense / Orbbec），只需在 `camera.py` 里加一个提供
`read() / release() / info()` 的类，`main.py` 改一行即可。文件末尾已附
RealSense / Orbbec 的示例骨架代码。

---

## 六、性能实测

| 指标 | 值 |
|------|-----|
| **单帧推理耗时** | **约 12.6 ms**（CPU，1280×720）→ 推理能力 ≈ **79 FPS** |
| 端到端帧率 | **约 20 FPS**（1280×720）—— 瓶颈是**摄像头读取阻塞**（30fps 读一帧要 33ms），不是推理 |
| 开 `--auto-roi` 时 | 约 17 FPS（检不到手时会多试几个区域；已内置"每 3 帧才试一次"的节流）|
| 模型大小 | 7.8 MB |
| 依赖 | mediapipe + opencv + numpy + Pillow，**无需 GPU** |

**想要更高帧率**：
```bat
python main.py --width 640 --height 480      :: 降到 640x480，端到端可到 30 FPS
python main.py --no-bars                     :: 少画一点
```

---

## 七、常见问题

| 问题 | 解决 |
|------|------|
| **打不开摄像头** | 关掉微信/QQ/浏览器/会议软件（会占用摄像头）；换 `--camera 1`；检查 Windows 隐私设置 |
| **检不到手** | ① 让手在画面里**大一点**（MediaPipe 需要手占较大比例）② `--auto-roi` ③ 降低 `--det-conf 0.2` ④ 换 `--roi` 限定区域 |
| **中文显示成方块** | 系统缺中文字体。本机实测用 `C:\Windows\Fonts\msyh.ttc` 正常 |
| **延迟大** | 降低分辨率 `--width 640 --height 480`；关闭 `--panel` |
| **手指角度看着不对** | 先用 `python test_angle.py` 验证公式（应输出"✅ 角度公式正确"） |
| **文件名含中文读不了** | 已内置 `imread_u/imwrite_u` 用 `np.fromfile + imdecode` 绕开 Windows 的 cv2 中文路径限制 |
| **`run.bat` 报"不是内部或外部命令"** | 批处理文件里**不能有中文**（cmd.exe 按 ANSI 码页解析 .bat）。本仓库的 `run.bat` 已改为**纯 ASCII + CRLF**，如你手动编辑过请保持这个格式 |
| **`run.bat` 报 `errorlevel=9009`** | 9009 = "命令找不到"，即 **cmd 里找不到 `python`**。常见于 conda 装了但 `python.exe` 所在目录不在系统 PATH（conda 只往 PATH 放了 `condabin`/`Scripts`/`Library\bin`，**没放根目录**）。**本项目 `run.bat` 已内置自动搜索**（见下），直接双击即可 |
| **`run.bat` 说"missing required packages"** | `where python` 命中了**微软商店的 Python 存根** —— `%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe`。它是个 0 字节的假 python（双击会弹商店），当然没有 cv2/mediapipe。**本项目已自动跳过任何路径含 `WindowsApps` 的候选**，并会**逐个候选试 import**，只接受真正装好依赖的那个 |
| **仍然找不到 Python** | 用记事本打开 `run.bat`，找到 `REM set "PY=D:\pkg\miniconda3\python.exe"`，**删掉 `REM `** 并改成你的真实路径 |

**`run.bat` 的 Python 搜索顺序**（逐个用 `import cv2, mediapipe, numpy, PIL` 验证，取第一个通过的）：
```
① PATH 里的 python（跳过 WindowsApps 存根）
② 常见 conda / Python 安装位置（D:\pkg\miniconda3 等 12 处）
③ %LOCALAPPDATA%\Programs\Python\Python3xx
④ conda info --base
⑤ py 启动器
```


---

## 八、原理与已知限制

### 用的什么
- **MediaPipe HandLandmarker（Tasks API）** —— mediapipe 1.0+ 已移除 `mp.solutions`，
  本代码用的是新的 `mediapipe.tasks.python.vision.HandLandmarker`。
- 输出 3 组数据：
  - `hand_landmarks`：21 点归一化 2D 坐标（绘图用）
  - `hand_world_landmarks`：21 点**手部度量 3D 坐标**（算角度用 ⭐）
  - `handedness`：左右手 + 置信度

### 已知限制
| 限制 | 说明 |
|------|------|
| **手太小会检不到** | MediaPipe 需要手占画面较大比例。实测 EgoTactile 广角帧（手很小）全幅检不到，裁剪后可检到 → 固定机位请**把手拍大**，或用 `--roi / --auto-roi` |
| 自遮挡 | 手指互相挡住时，被挡关节的估计会退化 |
| 极端角度 | 手背对相机、手指指向相机时精度下降 |
| 无真实尺度 | 关节角是**相对量**，不受距离影响；但若要"手在相机坐标系的位置/朝向"，需要深度相机 |
| 世界坐标非严格度量 | MediaPipe 的 world landmarks 是学习出来的、近似度量，不是深度测量 |

### 想更准 → 换深度相机
有了深度后：**2D 关键点 + 对齐后的深度图 → 真 3D 关键点**，
关节角精度和尺度都会更好。`camera.py` 里已预留接口。
