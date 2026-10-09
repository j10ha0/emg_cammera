"""visualize.py — 实时可视化

⚠️ OpenCV 的 putText **不支持中文**（会显示成 ???）。
   所以中文文本用 PIL + 系统字体渲染，数字/英文仍用 cv2.putText（更快）。
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from joint_angle import HAND_CONNECTIONS, MCPS, TIPS, HandState, gesture_hint

# ---------------------------------------------------------------- 配色（BGR）
C_WRIST = (0, 215, 255)
C_THUMB = (80, 80, 255)
C_INDEX = (80, 255, 80)
C_MIDDLE = (255, 200, 60)
C_RING = (255, 120, 200)
C_PINKY = (200, 255, 120)
FINGER_COLORS = [C_THUMB, C_INDEX, C_MIDDLE, C_RING, C_PINKY]
C_LM = (245, 245, 245)
C_TXT = (255, 255, 255)
C_DIM = (170, 170, 170)
C_OK = (120, 255, 120)
C_WARN = (60, 200, 255)
PANEL_BG = (28, 28, 32)


# ---------------------------------------------------------------- 中文字体
class CNFont:
    """自动查找系统中文字体；找不到则回退（中文会显示成方块）"""

    CANDIDATES = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]

    def __init__(self):
        self.path = next((p for p in self.CANDIDATES if os.path.isfile(p)), None)
        self._cache: dict[int, ImageFont.FreeTypeFont] = {}

    def get(self, size: int):
        if size not in self._cache:
            if self.path:
                try:
                    self._cache[size] = ImageFont.truetype(self.path, size)
                except Exception:
                    self._cache[size] = ImageFont.load_default()
            else:
                self._cache[size] = ImageFont.load_default()
        return self._cache[size]

    @property
    def ok(self) -> bool:
        return self.path is not None


_CN = CNFont()


def cn_panel(lines: list[tuple[str, int, tuple[int, int, int]]],
             size: tuple[int, int], bg=PANEL_BG, pad: int = 12,
             line_gap: int = 8) -> np.ndarray:
    """用 PIL 把多行文字渲染成一张 BGR 图片

    lines : [(文本, 字号, BGR颜色), ...]
    """
    w, h = size
    img = Image.new("RGB", (w, h), (bg[2], bg[1], bg[0]))
    d = ImageDraw.Draw(img)
    y = pad
    for text, size_px, color in lines:
        f = _CN.get(size_px)
        d.text((pad, y), text, font=f, fill=(color[2], color[1], color[0]))
        # 估算行高
        try:
            bbox = d.textbbox((pad, y), "汉Ag", font=f)
            lh = bbox[3] - bbox[1]
        except Exception:
            lh = size_px
        y += int(lh * 1.55) + line_gap
        if y > h - pad:
            break
    arr = np.asarray(img, dtype=np.uint8)          # RGB
    return arr[:, :, ::-1].copy()                  # -> BGR


def blend_panel(frame: np.ndarray, panel: np.ndarray, pos=(0, 0), alpha=0.82):
    """把面板半透明叠加到 frame 上"""
    x, y = pos
    ph, pw = panel.shape[:2]
    H, W = frame.shape[:2]
    x2, y2 = min(W, x + pw), min(H, y + ph)
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    p = panel[:y2 - y, :x2 - x]
    cv2.addWeighted(roi, 1 - alpha, p, alpha, 0, dst=roi)


# ---------------------------------------------------------------- 骨架
def draw_skeleton(frame: np.ndarray, lm2d: np.ndarray, thickness: int = 3):
    """画骨架连线 + 关键点"""
    H, W = frame.shape[:2]
    pts = np.stack([lm2d[:, 0] * W, lm2d[:, 1] * H], axis=1).astype(np.int32)

    # 按手指上色：拇指 0-4，食指 5-8，中指 9-12，无名 13-16，小指 17-20
    def color_for(a, b):
        for k, (s, e) in enumerate([(0, 4), (0, 8), (5, 12), (9, 16), (13, 20)]):
            if (a in range(1, 5) and k == 0) or (a in range(5, 9) and k == 1) \
               or (a in range(9, 13) and k == 2) or (a in range(13, 17) and k == 3) \
               or (a in range(17, 21) and k == 4):
                return FINGER_COLORS[k]
        return (200, 200, 200)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color_for(a, b), thickness, cv2.LINE_AA)

    for i, p in enumerate(pts):
        if i == 0:
            cv2.circle(frame, tuple(p), 8, C_WRIST, -1, cv2.LINE_AA)
        elif i in TIPS:
            cv2.circle(frame, tuple(p), 6, (255, 255, 255), -1, cv2.LINE_AA)
        elif i in MCPS:
            cv2.circle(frame, tuple(p), 5, (0, 220, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, tuple(p), 3, C_LM, -1, cv2.LINE_AA)
    return pts


def draw_angle_labels(frame: np.ndarray, pts: np.ndarray, st: HandState,
                      min_draw: float = 8.0):
    """在每个关节旁边标出角度（度）。只在角度可用时画。"""
    for fi, (f, defn) in enumerate(zip(st.fingers, [0, 1, 2, 3, 4])):
        # 关节位置：拇指标注在 ip(3)，其余标注在 pip
        jidx = 3 if f.name_en == "Thumb" else 6 + 4 * (defn - 1)
        p = pts[jidx]
        labels = []
        if not np.isnan(f.mcp):
            labels.append(("M", f.mcp))
        if not np.isnan(f.pip):
            labels.append(("P", f.pip))
        if not np.isnan(f.dip):
            labels.append(("D", f.dip))
        dx = 14
        # 关节角文本放在关节点右上
        y = p[1] - 10 - 16 * len(labels)
        for tag, val in labels:
            if val < min_draw:
                continue
            txt = f"{val:.0f}"
            org = (p[0] + dx, y)
            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        FINGER_COLORS[fi], 1, cv2.LINE_AA)
            y += 16
    return frame


# ---------------------------------------------------------------- 侧栏面板
def build_panel(st: HandState, fps: float, extra: list[str] | None = None,
                panel_w: int = 330, panel_h: int = 640) -> np.ndarray:
    """构造侧栏文字面板（含中文）"""
    side = "右手" if st.is_right else "左手"
    lines: list[tuple[str, int, tuple[int, int, int]]] = []
    lines.append((f"手别：{side}    姿态：{gesture_hint(st)}", 22, C_OK))
    lines.append((f"FPS {fps:5.1f}", 18, C_DIM))
    lines.append(("─" * 22, 14, (90, 90, 90)))
    lines.append(("关节角度（度）", 20, C_TXT))
    lines.append(("            MCP   PIP   DIP", 16, C_DIM))
    for f in st.fingers:
        m = "  -  " if np.isnan(f.mcp) else f"{f.mcp:5.0f}"
        p = "  -  " if np.isnan(f.pip) else f"{f.pip:5.0f}"
        d = "  -  " if np.isnan(f.dip) else f"{f.dip:5.0f}"
        lines.append((f"{f.name_cn:<4}{m} {p} {d}", 17, C_TXT))
    lines.append(("─" * 22, 14, (90, 90, 90)))
    lines.append(("张开度（度）", 20, C_TXT))
    for k, v in st.spread.items():
        lines.append((f"  {k:<14}{v:6.1f}", 16, C_TXT))
    lines.append(("─" * 22, 14, (90, 90, 90)))
    lines.append(("其他", 20, C_TXT))
    lines.append((f"  拇指-食指对合 {st.opposition:5.2f}  (越小越贴合)", 15, C_TXT))
    lines.append((f"  掌长 {st.palm_length*1000:5.1f} mm   掌宽 {st.palm_width*1000:5.1f} mm",
                  15, C_TXT))
    lines.append((f"  四指平均弯曲 {st.total_curl:.2f}", 15, C_TXT))
    if extra:
        lines.append(("─" * 22, 14, (90, 90, 90)))
        for t in extra:
            lines.append((t, 15, C_WARN))
    return cn_panel(lines, (panel_w, panel_h))


def draw_curl_bars(frame: np.ndarray, st: HandState, origin=(16, 16),
                   bar_w: int = 150, bar_h: int = 14, gap: int = 6):
    """画手指弯曲度条形（0~1）"""
    x0, y0 = origin
    for i, f in enumerate(st.fingers):
        y = y0 + i * (bar_h + gap)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (60, 60, 60), -1)
        w = int(bar_w * float(np.clip(f.curl, 0, 1)))
        cv2.rectangle(frame, (x0, y), (x0 + w, y + bar_h), FINGER_COLORS[i], -1)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (120, 120, 120), 1)
        cv2.putText(frame, f.name_en[:5], (x0 + bar_w + 8, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, texts: list[str], origin=(16, 16)):
    x, y = origin
    for t in texts:
        cv2.putText(frame, t, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TXT, 1, cv2.LINE_AA)
        y += 22
    return frame
