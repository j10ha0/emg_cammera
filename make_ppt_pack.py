"""make_ppt_pack.py — 一键生成 PPT 展示图包（3 张图）

产物：
  fig1_system.jpg     系统效果总览（手部骨架 + 关节角 + 数据面板，模拟实时界面）
  fig2_poses.jpg      不同手型/物体 的标注拼图（3x3）
  fig3_robustness.jpg 关节角可靠性分析（抗噪能力条形图）

用法:
    python make_ppt_pack.py                      # 用 nc_src/
    python make_ppt_pack.py --src ppt_src --out ppt_figures
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
from make_ppt_figures import best_hand_crop, imread_u, imwrite_u
from visualize import (CNFont, blend_panel, build_panel, draw_angle_labels,
                       draw_curl_bars, draw_hud, draw_skeleton)

HERE = os.path.dirname(os.path.abspath(__file__))
_CN = CNFont()
BG = (247, 247, 250)
FG = (24, 24, 30)
FG2 = (92, 92, 104)


def newcanvas(w, h, title=None, subtitle=None):
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    y = 18
    if title:
        d.text((28, y), title, font=_CN.get(36), fill=FG)
        y += 52
    if subtitle:
        d.text((28, y), subtitle, font=_CN.get(19), fill=FG2)
        y += 36
    return im, d, y + 6


# ============================================================ fig1
def fig1_system(rows, out, size=(1600, 1000)):
    """模拟"实时程序界面"：手部特写 + 骨架/角度 + 右侧数据面板 + 弯曲度条形"""
    name, crop, hands = rows[0]
    h0 = max(hands, key=lambda z: z.score)
    lm = h0.lm3d if np.any(h0.lm3d) else h0.lm2d
    st = compute_hand_state(lm, is_right=h0.is_right)

    vis = crop.copy()
    H, W = vis.shape[:2]
    # 缩放到合适大小
    tw = 1180
    vis = cv2.resize(vis, (tw, int(H * tw / W)), interpolation=cv2.INTER_CUBIC)
    H, W = vis.shape[:2]

    pts = draw_skeleton(vis, h0.lm2d, thickness=3)
    draw_angle_labels(vis, pts, st, min_draw=5.0)
    draw_hud(vis, [f"{'Right' if st.is_right else 'Left'}   curl {st.total_curl:.2f}"],
             origin=(16, 30))
    draw_curl_bars(vis, st, origin=(16, H - 6 * 26 - 14), bar_w=170)
    panel = build_panel(st, 0.0, panel_w=340, panel_h=min(H, 660))
    blend_panel(vis, panel, pos=(W - 340, 0), alpha=0.80)

    im, d, y = newcanvas(size[0], size[1],
                         "图1  实时手部关节角度标注 —— 程序界面",
                         "固定机位 USB 摄像头 → MediaPipe(21 关键点) → 关节角(MCP/PIP/DIP) + 张开度 + 拇指-食指对合；约 20 FPS，无需 GPU")
    pil = Image.fromarray(vis[:, :, ::-1])
    maxw = size[0] - 56
    if pil.width > maxw:
        pil = pil.resize((maxw, int(pil.height * maxw / pil.width)), Image.LANCZOS)
    im.paste(pil, (28, y))
    yy = y + pil.height + 12
    d.text((28, yy), f"示例：{name}    手别 {'右手' if st.is_right else '左手'}    "
                     f"检测置信度 {h0.score:.2f}    姿态 {gesture_hint(st)}",
           font=_CN.get(21), fill=FG)
    im.save(out, quality=93)
    print(f"  → {out}  ({im.width}x{im.height})")


# ============================================================ fig2
def fig2_poses(rows, out, cols=3, max_n=9, tile_w=600):
    use = rows[:max_n]
    rn = math.ceil(len(use) / cols)
    tiles = []
    caps = []
    for name, crop, hands in use:
        h0 = max(hands, key=lambda z: z.score)
        lm = h0.lm3d if np.any(h0.lm3d) else h0.lm2d
        st = compute_hand_state(lm, is_right=h0.is_right)
        vis = crop.copy()
        pts = draw_skeleton(vis, h0.lm2d, thickness=3)
        draw_angle_labels(vis, pts, st, min_draw=5.0)
        f = st.fingers[1]
        im = Image.fromarray(vis[:, :, ::-1])
        im = im.resize((tile_w, int(im.height * tile_w / im.width)), Image.LANCZOS)
        tiles.append(im)
        caps.append((name, st, h0.score, f))

    caph = 78
    th = max(t.height for t in tiles)
    pad, top = 12, 100
    W = cols * tile_w + (cols + 1) * pad
    Hh = top + rn * (th + caph + pad) + pad
    im, d, _ = newcanvas(W, Hh,
                         "图2  不同物体 / 不同手型 的关节角标注结果",
                         "每格：彩色骨架 + 各关节角度（M/P/D）；底部为物体名、手别、检测置信度、食指 MCP/PIP/DIP 与平均弯曲度")
    for k, (t, (name, st, sc, f)) in enumerate(zip(tiles, caps)):
        r_, c_ = divmod(k, cols)
        x = pad + c_ * (tile_w + pad)
        y = top + r_ * (th + caph + pad)
        im.paste(t, (x, y))
        bx = x
        by = y + th
        # 说明条
        d.rectangle([bx, by, bx + tile_w, by + caph], fill=(236, 236, 242))
        d.text((bx + 10, by + 8), f"{name}", font=_CN.get(21), fill=FG)
        d.text((bx + 10, by + 36),
               f"{'右手' if st.is_right else '左手'}  conf {sc:.2f}   "
               f"食指 {f.mcp:.0f}/{f.pip:.0f}/{f.dip:.0f}°   curl {st.total_curl:.2f}",
               font=_CN.get(17), fill=FG2)
    im.save(out, quality=92)
    print(f"  → {out}  ({im.width}x{im.height})")


# ============================================================ fig3
def fig3_robustness(out, size=(1500, 860)):
    """关节角抗噪能力（复用 test_angle_robustness 的原理现算）"""
    rng = np.random.default_rng(0)
    lm0 = np.zeros((21, 3), float)
    lm0[0] = [0, 0, 0]
    for base, x in {5: 0.030, 9: 0.055, 13: 0.078, 17: 0.098}.items():
        lm0[base] = [x, 0.085, 0.000]
        lm0[base + 1] = [x, 0.120, 0.018]
        lm0[base + 2] = [x, 0.140, 0.045]
        lm0[base + 3] = [x, 0.148, 0.072]
    lm0[1] = [-0.030, 0.030, 0.0]
    lm0[2] = [-0.058, 0.052, 0.010]
    lm0[3] = [-0.080, 0.074, 0.028]
    lm0[4] = [-0.095, 0.090, 0.048]

    sig_levels = [1.0, 2.0, 3.0]                    # mm
    series = {"食指 MCP": [], "食指 PIP": [], "食指 DIP": [],
              "拇指 MCP": [], "拇指 IP": []}
    for s_mm in sig_levels:
        sig = s_mm / 1000.0
        acc = {k: [] for k in series}
        for _ in range(300):
            st = compute_hand_state(lm0 + rng.normal(0, sig, lm0.shape))
            acc["食指 MCP"].append(st.fingers[1].mcp)
            acc["食指 PIP"].append(st.fingers[1].pip)
            acc["食指 DIP"].append(st.fingers[1].dip)
            acc["拇指 MCP"].append(st.fingers[0].mcp)
            acc["拇指 IP"].append(st.fingers[0].pip)
        for k in series:
            series[k].append(float(np.nanstd(acc[k])))

    im, d, y = newcanvas(size[0], size[1],
                         "图3  关节角可靠性分析（关键点加噪后的角度标准差）",
                         "MediaPipe 的 3D 关键点存在毫米级误差。给关键点加不同幅度高斯噪声，重复 300 次计算角度 → σ 越小越可信")
    left, right = 210, size[0] - 70
    plot_h = 430
    top = y + 24
    ymax = max(max(v) for v in series.values()) * 1.15
    colors = {"食指 MCP": (80, 170, 90), "食指 PIP": (235, 170, 60),
              "食指 DIP": (215, 80, 80), "拇指 MCP": (90, 140, 220),
              "拇指 IP": (170, 110, 200)}

    # 轴
    d.line([left, top, left, top + plot_h], fill=(150, 150, 160), width=2)
    d.line([left, top + plot_h, right, top + plot_h], fill=(150, 150, 160), width=2)
    # 网格 + y 轴刻度
    for gy in range(0, 6):
        v = ymax * gy / 5
        yy = top + plot_h - int(plot_h * gy / 5)
        d.line([left, yy, right, yy], fill=(222, 222, 230), width=1)
        d.text((left - 62, yy - 10), f"{v:.0f}", font=_CN.get(17), fill=FG2)
    d.text((left - 62, top - 30), "σ (°)", font=_CN.get(18), fill=FG)

    # 每组三条柱
    gw = (right - left) / len(sig_levels)
    bw = gw / (len(series) + 1.6)
    for gi, s_mm in enumerate(sig_levels):
        gx = left + gi * gw + gw * 0.12
        for si, (k, vals) in enumerate(series.items()):
            h = int(plot_h * vals[gi] / ymax)
            x0 = int(gx + si * bw)
            col = colors[k]
            d.rectangle([x0, top + plot_h - h, x0 + int(bw * 0.82), top + plot_h],
                        fill=col)
            d.text((x0 - 4, top + plot_h - h - 22), f"{vals[gi]:.1f}",
                   font=_CN.get(15), fill=FG)
        d.text((int(left + gi * gw + gw * 0.22), top + plot_h + 14),
               f"关键点噪声 σ = {s_mm:.0f} mm", font=_CN.get(19), fill=FG)

    # 图例
    lx, ly = size[0] - 300, y + 10
    for i, (k, c) in enumerate(colors.items()):
        yy = ly + i * 26
        d.rectangle([lx, yy, lx + 20, yy + 16], fill=c)
        d.text((lx + 28, yy - 2), k, font=_CN.get(18), fill=FG)

    ty = top + plot_h + 60
    d.text((28, ty), "结论：MCP 由较长骨段决定 → 抗噪最好（主要可用指标）；"
                     "PIP 次之；DIP 由很短末节决定 → 对关键点误差最敏感，仅作参考。",
           font=_CN.get(20), fill=FG)
    d.text((28, ty + 34), "报告建议：以 MCP + PIP 作为主要评价指标，DIP 配合时序平滑使用。",
           font=_CN.get(20), fill=FG)
    im.save(out, quality=93)
    print(f"  → {out}  ({im.width}x{im.height})")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "nc_src"))
    ap.add_argument("--out", default=os.path.join(HERE, "ppt_figures"))
    ap.add_argument("--det-conf", type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(sum([glob.glob(os.path.join(a.src, e))
                        for e in ("*.jpg", "*.jpeg", "*.png")], []))
    print(f"源图 {len(files)} 张")

    est = HandPoseEstimator(num_hands=2, det_conf=a.det_conf, pres_conf=a.det_conf,
                            track_conf=a.det_conf, verbose=False)
    rows = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0].replace("_nc", "")
        img = imread_u(fp)
        if img is None:
            continue
        r = best_hand_crop(est, img)
        if r is None:
            print(f"  {name:<32} 未找到手")
            continue
        sc, crop, hands = r
        crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                          interpolation=cv2.INTER_CUBIC)
        rows.append((name, crop, hands))
        print(f"  {name:<32} ✓ 置信 {sc:.2f}")
    est.close()

    if not rows:
        print("没有可用结果")
        return
    rows.sort(key=lambda z: -max(h.score for h in z[2]))

    print("\n生成图包 →", a.out)
    fig1_system(rows, os.path.join(a.out, "fig1_system.jpg"))
    fig2_poses(rows, os.path.join(a.out, "fig2_poses.jpg"))
    fig3_robustness(os.path.join(a.out, "fig3_robustness.jpg"))
    print("\n完成。三张图可直接拖进 PPT。")


if __name__ == "__main__":
    main()
