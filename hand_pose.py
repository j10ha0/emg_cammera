"""hand_pose.py — MediaPipe HandLandmarker 封装（Tasks API）

注意：mediapipe 1.0+ **已移除 `mp.solutions`**，必须用新的 Tasks API。
本模块负责：
  · 首次运行时自动下载 hand_landmarker.task 模型（约 7.5 MB）
  · 逐帧推理，返回规范化 2D 关键点 + 世界(度量)3D 关键点 + 左右手
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def ensure_model(model_path: str | None = None) -> str:
    """模型不存在则自动下载，返回本地路径"""
    if model_path is None:
        model_path = os.path.join(DEFAULT_MODEL_DIR, "hand_landmarker.task")
    if os.path.isfile(model_path) and os.path.getsize(model_path) > 100_000:
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[hand_pose] 首次运行，正在下载模型 → {model_path}")
    tmp = model_path + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, model_path)
        print(f"[hand_pose] 模型下载完成 ({os.path.getsize(model_path)/1e6:.1f} MB)")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"模型下载失败：{e}\n"
            f"请手动下载后放到：{model_path}\n"
            f"下载地址：{MODEL_URL}"
        ) from e
    return model_path


@dataclass
class HandResult:
    """单只手的检测结果"""
    is_right: bool                  # True=右手
    score: float                    # 左右手判别置信度
    lm2d: np.ndarray                # (21, 3) 归一化坐标 x,y ∈[0,1]，z 相对深度
    lm3d: np.ndarray                # (21, 3) 世界坐标（米），以手几何中心为原点
    label: str = ""

    @property
    def side(self) -> str:
        return "Right" if self.is_right else "Left"


class HandPoseEstimator:
    """封装 HandLandmarker 的 VIDEO 模式，逐帧推理"""

    def __init__(self, model_path: str | None = None, num_hands: int = 2,
                 det_conf: float = 0.30, pres_conf: float = 0.30, track_conf: float = 0.30,
                 verbose: bool = True):
        self.model_path = ensure_model(model_path)
        base = mp_python.BaseOptions(model_asset_path=self.model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=pres_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self._t_ms = 0
        self._last_ts = -1
        if verbose:
            print(f"[hand_pose] HandLandmarker 就绪 (num_hands={num_hands})")

    # ------------------------------------------------------------- infer
    def detect(self, bgr_frame: np.ndarray, timestamp_ms: int | None = None,
               roi: tuple[float, float, float, float] | None = None) -> list[HandResult]:
        """输入 BGR 帧，返回每只手的 HandResult 列表

        roi : 可选，(x1, y1, x2, y2) 归一化 0~1 的检测区域。
              **当手在全幅里太小时**（如 ego 广角），限制 ROI 能让 MediaPipe 看清手。
              返回的关键点仍映射回【整幅图】的归一化坐标。
        """
        H, W = bgr_frame.shape[:2]
        ox = oy = 0
        work = bgr_frame
        if roi is not None:
            x1, y1, x2, y2 = [int(round(v * s)) for v, s in
                              zip(roi, (W, H, W, H))]
            x1, x2 = max(0, min(x1, W - 2)), max(2, min(x2, W))
            y1, y2 = max(0, min(y1, H - 2)), max(2, min(y2, H))
            if x2 - x1 >= 32 and y2 - y1 >= 32:
                work = np.ascontiguousarray(bgr_frame[y1:y2, x1:x2])
                ox, oy = x1, y1
        hw, hh = work.shape[1], work.shape[0]

        rgb = np.ascontiguousarray(work[:, :, ::-1])       # BGR -> RGB
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if timestamp_ms is None:
            # VIDEO 模式要求时间戳严格递增
            self._t_ms = max(self._t_ms + 1, self._last_ts + 1)
            ts = self._t_ms
        else:
            ts = int(timestamp_ms)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
        self._last_ts = ts

        res = self.landmarker.detect_for_video(mp_img, ts)
        out: list[HandResult] = []
        if not res.hand_landmarks:
            return out

        for i, lms in enumerate(res.hand_landmarks):
            lm2d = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            if roi is not None:
                # 从裁剪区域坐标映射回整幅图的归一化坐标
                lm2d[:, 0] = (lm2d[:, 0] * hw + ox) / W
                lm2d[:, 1] = (lm2d[:, 1] * hh + oy) / H
            wl = res.hand_world_landmarks[i] if res.hand_world_landmarks else None
            lm3d = (np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float64)
                    if wl else np.zeros((21, 3)))
            side, score = "Right", 0.0
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                cat = res.handedness[i][0]
                side = cat.category_name or "Right"
                score = float(cat.score or 0.0)
            out.append(HandResult(is_right=(side.lower().startswith("r")),
                                  score=score, lm2d=lm2d, lm3d=lm3d, label=side))
        return out

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
