# 数据来源与许可声明

本仓库代码为**自研**；仓库中附带的部分**测试图片**来自公开数据集，仅为复现展示图之用。
这些数据**不用于商业用途**，并在此标注来源与许可。

---

## 1. 测试图片（`test_images/`、`test_images2/`、`nc_src/`、`ppt_src/`）

| 来源 | 说明 | 许可 |
|------|------|------|
| **EgoTactile** | 头戴/颈戴 egocentric 视频，绿幕背景，63 个日常物体抓握。<br>论文：*EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video*, ICML 2026 spotlight | **CC BY-NC 4.0**<br>（署名 + 非商业）|
| **EgoDex** | Apple Vision Pro 采集的 egocentric 手部操作数据 | 见数据集官网 |

**链接**：
- EgoTactile 项目页：https://egotactile.github.io/
- EgoTactile 数据：https://huggingface.co/datasets/HustleHard/EgoTactile
- EgoTactile 论文：https://arxiv.org/abs/2606.09243
- EgoDex：https://github.com/apple/ml-egodex

> ⚠️ 这些图片来自 egocentric 广角视频，**手在画面中占比偏小**，属于 MediaPipe 的困难样本
> （本仓库用它们来展示"最难的情况下也能检出"）。实际使用时请把相机取景调好，让手占画面较大比例。

**如需重新获取原始数据**，请从上述官方渠道下载；本仓库不提供完整数据集。

---

## 2. 模型

| 组件 | 来源 | 许可 |
|------|------|------|
| **MediaPipe HandLandmarker** | Google MediaPipe | Apache-2.0 |
| 模型文件 `hand_landmarker.task` | `storage.googleapis.com/mediapipe-models`（首次运行自动下载，**不入库**）| Apache-2.0 |

---

## 3. 代码依赖

| 包 | 许可 |
|----|------|
| opencv-python | Apache-2.0 |
| mediapipe | Apache-2.0 |
| numpy | BSD-3-Clause |
| Pillow | HPND |

---

## 4. 引用建议

如果本仓库的代码或展示图对你的工作有帮助，建议引用以下工作：

```bibtex
@inproceedings{egotactile2026,
  title     = {EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video},
  booktitle = {ICML},
  year      = {2026}
}
```
