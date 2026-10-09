"""make_ppt_figures.py — 生成 PPT 用的展示图

做两件事:
  ① 对每张源图，自动搜索「最佳手部特写」（网格搜索 + 选置信度最高的窗口）
     → 跑骨架/角度标注 → 存成单张图
  ② 把成功的结果拼成一张 montage（带中文标题），PPT 直接用

用法:
    python make_ppt_figures.py                      # 处理 ppt_src/
    python make_ppt_figures.py --src 别的目录 --out ppt_out
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hand_pose import HandPoseEstimator
from joint_angle import compute_hand_state, gesture_hint
from visualize import CNFont, draw_angle_labels, draw_skeleton

HERE = os.path.dirname(os.path.abspath(__file__))
_CN = CNFont()


# ------------------------------------------------------------------ 工具
def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img):
    ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".jpg", img)
    if ok:
        buf.tofile(p)
    return ok


def draw_caption(img, lines, height=74):
    """在图片底部加一条说明栏（中文，用 PIL 渲染）"""
    H, W = img.shape[:2]
    bar = np.full((height, W, 3), 32, np.uint8)
    pil = Image.new("RGB", (W, height), (32, 32, 32))
    d = ImageDraw.Draw(pil)
    y = 8
    for text, size, bgr in lines:
        f = _CN.get(size)
        d.text((14, y), text, font=f, fill=(bgr[2], bgr[1], bgr[0]))
        try:
            bb = d.textbbox((14, y), "漢Ag", font=f)
            lh = bb[3] - bb[1]
        except Exception:
            lh = size
        y += int(lh * 1.5)
    bar = np.asarray(pil, np.uint8)[:, :, ::-1].copy()
    return np.vstack([img, bar])


# ------------------------------------------------------------------ 搜最佳特写
def best_hand_crop(est, img, scales=(0.85, 0.70, 0.55), step=0.2, early=0.90):
    """网格搜索：返回 (score, crop, hands) 取置信度最高的窗口"""
    H, W = img.shape[:2]
    best = None
    for s in scales:
        cw, ch = int(W * s), int(H * s)
        if cw < 160 or ch < 160:
            continue
        ys = np.arange(0, max(1, H - ch) + 1, max(1, int((H - ch) * step) or 1))
        xs = np.arange(0, max(1, W - cw) + 1, max(1, int((W - cw) * step) or 1))
        for y in ys:
            for x in xs:
                crop = img[int(y):int(y) + ch, int(x):int(x) + cw]
                if crop.shape[0] < 160 or crop.shape[1] < 160:
                    continue
                hands = est.detect(crop)
                if hands:
                    sc = max(h.score for h in hands)
                    if best is None or sc > best[0]:
                        best = (sc, crop, hands)
                    if sc >= early:
                        return best
    return best


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "ppt_src"))
    ap.add_argument("--out", default=os.path.join(HERE, "ppt_out"))
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="低于该置信度的结果不采用")
    ap.add_argument("--up", type=int, default=2, help="裁剪后放大倍数")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(sum([glob.glob(os.path.join(a.src, e))
                        for e in ("*.jpg", "*.jpeg", "*.png")], []))
    print(f"源图 {len(files)} 张 → {a.out}")

    est = HandPoseEstimator(num_hands=2, det_conf=0.25, pres_conf=0.25,
                            track_conf=0.25, verbose=False)
    rows = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        img = imread_u(fp)
        if img is None:
            print(f"  {name:<34} 读不了")
            continue
        r = best_hand_crop(est, img)
        if r is None or r[0] < a.min_score:
            print(f"  {name:<34} 未找到手")
            continue
        score, crop, hands = r

        # 放大以便阅读
        if a.up > 1:
            crop = cv2.resize(crop, (crop.shape[1] * a.up, crop.shape[0] * a.up),
                              interpolation=cv2.INTER_CUBIC)

        h0 = max(hands, key=lambda z: z.score)
        lm = h0.lm3d if np.any(h0.lm3d) else h0.lm2d
        st = compute_hand_state(lm, is_right=h0.is_right)

        vis = crop.copy()
        pts = draw_skeleton(vis, h0.lm2d, thickness=3)
        draw_angle_labels(vis, pts, st, min_draw=5.0)

        idx = st.fingers[1]
        cap = [
            (f"{name}   手别: {'右手' if st.is_right else '左手'}"
             f"   姿态: {gesture_hint(st)}   置信度 {score:.2f}", 22, (120, 255, 120)),
            (f"食指 MCP {idx.mcp:.0f}°   PIP {idx.pip:.0f}°   DIP {idx.dip:.0f}°"
             f"      平均弯曲度 {st.total_curl:.2f}"
             f"      拇指-食指对合 {st.opposition:.2f}", 18, (235, 235, 235)),
        ]
        out_img = draw_caption(vis, cap, height=78)
        op = os.path.join(a.out, f"{name}_annotated.jpg")
        imwrite_u(op, out_img)
        rows.append((name, op, st, score))
        print(f"  {name:<34} ✓ 置信 {score:.2f}  "
              f"食指MCP/PIP/DIP {idx.mcp:.0f}/{idx.pip:.0f}/{idx.dip:.0f}  "
              f"弯曲 {st.total_curl:.2f}  {gesture_hint(st)}")

    est.close()

    # ---------------- montage ----------------
    if rows:
        cols = 3
        rows_n = math.ceil(len(rows) / cols)
        tw = 620
        tiles = []
        for name, op, st, score in rows:
            im = Image.open(op).convert("RGB")
            im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
            tiles.append(im)
        cw = tw
        chh = max(t.height for t in tiles)
        pad, top = 10, 92
        sheet = Image.new("RGB", (cols * cw + (cols + 1) * pad,
                                  top + rows_n * chh + (rows_n + 1) * pad),
                          (245, 245, 248))
        d = ImageDraw.Draw(sheet)
        f1 = _CN.get(34)
        f2 = _CN.get(20)
        d.text((pad + 4, 14), "实时手部关节角度标注 —— 不同抓握手型效果", font=f1, fill=(20, 20, 25))
        d.text((pad + 4, 58), f"MediaPipe HandLandmarker  ·  共 {len(rows)} 组  ·  "
                             f"角度由手部度量 3D 坐标计算（与相机距离/角度无关）",
               font=f2, fill=(90, 90, 100))
        for k, t in enumerate(tiles):
            r_, c_ = divmod(k, cols)
            sheet.paste(t, (pad + c_ * (cw + pad), top + pad + r_ * (chh + pad)))
        mp = os.path.join(a.out, "montage_poses.jpg")
        sheet.save(mp, quality=92)
        print(f"\n✓ montage → {mp}   ({sheet.width}x{sheet.height})")
    print(f"成功 {len(rows)}/{len(files)}")


if __name__ == "__main__":
    main()
