"""capture_poses.py — 从摄像头交互式采集「多手型」展示图

用途：做 PPT 时现场摆几个手型，各抓一张。
程序会持续显示画面，你摆好姿势后按 SPACE 抓拍，按 q 结束。

用法:
    python capture_poses.py                 # 摄像头0
    python capture_poses.py --camera 1 --labels 张开 握拳 捏合 三指 五指抓 侧捏
    python capture_poses.py --headless 20   # 无窗口：自动抓 N 张置信度最高的
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from camera import Camera
from hand_pose import HandPoseEstimator
from joint_angle import compute_hand_state, gesture_hint
from visualize import (blend_panel, build_panel, draw_angle_labels,
                       draw_curl_bars, draw_hud, draw_skeleton)

HERE = os.path.dirname(os.path.abspath(__file__))


def imwrite_u(p, img):
    ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".jpg", img)
    if ok:
        buf.tofile(p)
    return ok


def annotate(frame, est, det_conf):
    """返回 (annotated, primary_state)"""
    hands = est.detect(frame)
    primary = None
    for h in hands:
        lm = h.lm3d if np.any(h.lm3d) else h.lm2d
        st = compute_hand_state(lm, is_right=h.is_right)
        if primary is None:
            primary = (st, h)
        pts = draw_skeleton(frame, h.lm2d)
        draw_angle_labels(frame, pts, st)
    H, W = frame.shape[:2]
    if primary is not None:
        st, h = primary
        draw_hud(frame, [f"{'Right' if st.is_right else 'Left'}  curl {st.total_curl:.2f}"],
                 origin=(16, 26))
        draw_curl_bars(frame, st, origin=(16, H - 6 * 26 - 12))
        pw = 340
        panel = build_panel(st, 0.0, panel_w=pw, panel_h=min(H, 660))
        blend_panel(frame, panel, pos=(W - pw, 0), alpha=0.80)
    return frame, primary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default=os.path.join(HERE, "ppt_out_camera"))
    ap.add_argument("--labels", nargs="*", default=[
        "张开", "握拳", "捏合", "三指抓", "五指抓", "侧捏"])
    ap.add_argument("--det-conf", type=float, default=0.30)
    ap.add_argument("--headless", type=int, default=0,
                    help="无窗口：自动处理 N 帧，保留置信度最高的几张")
    ap.add_argument("--keep", type=int, default=6, help="headless 保留几张")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cam = Camera(index=a.camera, width=a.width, height=a.height, fps=30, mirror=True)
    print("[cam]", cam.info())
    est = HandPoseEstimator(num_hands=2, det_conf=a.det_conf, pres_conf=a.det_conf,
                            track_conf=a.det_conf)
    print(f"[out] {a.out}")

    # ---------------- headless：自动抓置信度最高的 N 张 ----------------
    if a.headless:
        best = []          # (score, annotated, state)
        for i in range(a.headless):
            ok, frame = cam.read()
            if not ok:
                break
            vis, primary = annotate(frame.copy(), est, a.det_conf)
            if primary is not None:
                st, h = primary
                best.append((h.score, vis, st))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.headless}  已收集 {len(best)}")
        best.sort(key=lambda z: -z[0])
        for k, (sc, vis, st) in enumerate(best[:a.keep]):
            p = os.path.join(a.out, f"pose_{k+1:02d}_score{sc:.2f}.jpg")
            imwrite_u(p, vis)
            print(f"  → {os.path.basename(p)}  "
                  f"食指PIP {st.fingers[1].pip:.0f}°  弯曲 {st.total_curl:.2f}  "
                  f"{gesture_hint(st)}")
        cam.release(); est.close()
        print(f"完成：{len(best)}/{a.headless} 帧检到手，存了 {min(len(best), a.keep)} 张")
        return

    # ---------------- 交互式：摆姿势按 SPACE 抓拍 ----------------
    print("\n操作：摆好手型 → 按 SPACE 抓拍 → 按 q 结束")
    print("期望的手型标签顺序：", " → ".join(a.labels), "\n")
    shot = 0
    tip = (f"SPACE = capture next pose   q = quit   "
           f"(next label: {a.labels[min(shot, len(a.labels)-1)]})")
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        vis, primary = annotate(frame.copy(), est, a.det_conf)
        cv2.putText(vis, tip, (16, cam.height - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, tip, (16, cam.height - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (120, 255, 255), 1, cv2.LINE_AA)
        if primary is None:
            cv2.putText(vis, "no hand", (16, cam.height - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("capture poses", vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            if primary is None:
                print("  [!] 当前没检到手，没抓拍")
                continue
            label = a.labels[min(shot, len(a.labels) - 1)]
            p = os.path.join(a.out, f"{shot+1:02d}_{label}.jpg")
            imwrite_u(p, vis)
            st = primary[0]
            print(f"  ✓ {os.path.basename(p)}   食指MCP/PIP/DIP "
                  f"{st.fingers[1].mcp:.0f}/{st.fingers[1].pip:.0f}/{st.fingers[1].dip:.0f}  "
                  f"弯曲 {st.total_curl:.2f}")
            shot += 1
            tip = (f"SPACE = capture next pose   q = quit   "
                   f"(next label: {a.labels[min(shot, len(a.labels)-1)]})")
            if shot >= len(a.labels):
                print("  已采集完所有标签，按 q 退出")

    cam.release(); est.close(); cv2.destroyAllWindows()
    print(f"\n共抓拍 {shot} 张 → {a.out}")


if __name__ == "__main__":
    main()
