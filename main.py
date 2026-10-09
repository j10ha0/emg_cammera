"""main.py — 实时摄像头手部关节角度标注

用法：
    python main.py                     # 默认摄像头 0，开窗口实时显示
    python main.py --camera 1          # 指定摄像头
    python main.py --width 1280 --height 720 --fps 30
    python main.py --no-mirror         # 不水平翻转
    python main.py --record            # 同时录制标注视频
    python main.py --log               # 同时把角度写入 CSV
    python main.py --headless 60       # 无窗口：处理 60 帧，存图到 out/（用于验证）

按键：
    q / ESC   退出
    s         截图
    m         切换镜像
    空格      暂停/继续
    r         开始/停止录制
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

from camera import Camera
from hand_pose import HandPoseEstimator
from joint_angle import compute_hand_state, gesture_hint
from visualize import (blend_panel, build_panel, draw_angle_labels,
                       draw_curl_bars, draw_hud, draw_skeleton, CNFont)

HERE = os.path.dirname(os.path.abspath(__file__))

# 手势提示的英文版（cv2.putText 不支持中文）
_HINT_EN = {
    "握拳 / 抓握": "FIST / GRASP",
    "张开 / 伸展": "OPEN / EXTENDED",
    "捏合 / 对指": "PINCH",
    "半握": "HALF-CLOSE",
    "中间状态": "INTERMEDIATE",
}


def gesture_hint_en(st) -> str:
    return _HINT_EN.get(gesture_hint(st), "?")


# ---------------------------------------------------------------- ROI 策略
# 当手在全幅里太小（广角 / 固定机位离得远）时，MediaPipe 检不到。
# 对策：限制检测区域（ROI），等价于把该区域放大后再检测。
AUTO_ROIS = [
    (0.25, 0.25, 0.75, 0.75),      # 中心 50%
    (0.20, 0.30, 0.80, 0.95),      # 中下 60%
    (0.40, 0.15, 0.95, 0.80),      # 右中
    (0.05, 0.15, 0.60, 0.80),      # 左中
]


def detect_hands(est: HandPoseEstimator, frame: np.ndarray,
                 roi=None, auto_roi: bool = False, ts=None):
    """带 ROI 回退的手部检测

    顺序：指定 ROI（若有）→ 全幅 → 自动候选 ROI 列表
    返回 (hands, used_roi)
    """
    if roi is not None:
        h = est.detect(frame, timestamp_ms=ts, roi=roi)
        if h:
            return h, roi
    h = est.detect(frame, timestamp_ms=ts)
    if h or not auto_roi:
        return h, None
    for cand in AUTO_ROIS:
        h = est.detect(frame, timestamp_ms=ts, roi=cand)
        if h:
            return h, cand
    return [], None


def parse_roi(s):
    """'0.2,0.2,0.8,0.9' -> (0.2, 0.2, 0.8, 0.9)"""
    if not s:
        return None
    try:
        v = tuple(float(x) for x in s.split(","))
        if len(v) != 4:
            raise ValueError
        return v
    except Exception:
        print("[main] --roi 格式应为 x1,y1,x2,y2（如 0.2,0.2,0.8,0.9），已忽略")
        return None


# ---------------------------------------------------------------- 中文路径
# ⚠️ Windows 上 cv2.imread / cv2.imwrite 处理不了中文路径，会返回 None。
#    用 np.fromfile + cv2.imdecode 绕开。
def imread_u(path: str):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path: str, img) -> bool:
    ext = os.path.splitext(path)[1] or ".jpg"
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return False


def parse_args():
    p = argparse.ArgumentParser(description="实时手部关节角度标注")
    p.add_argument("--camera", type=int, default=0, help="摄像头索引（默认 0）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-mirror", action="store_true", help="不水平翻转画面")
    p.add_argument("--num-hands", type=int, default=2)
    p.add_argument("--det-conf", type=float, default=0.30,
                   help="手部检测置信度阈值（低=更容易检到，但可能有误检）")
    p.add_argument("--pres-conf", type=float, default=0.30,
                   help="手存在置信度阈值")
    p.add_argument("--track-conf", type=float, default=0.30,
                   help="跟踪置信度阈值")
    p.add_argument("--panel", action="store_true", help="显示右侧数据面板（中文）")
    p.add_argument("--no-bars", action="store_true", help="不显示弯曲度条形")
    p.add_argument("--roi", default=None,
                   help="限定检测区域，归一化坐标 x1,y1,x2,y2（如 0.2,0.2,0.8,0.9）")
    p.add_argument("--auto-roi", action="store_true",
                   help="全幅检不到手时，自动尝试若干 ROI（广角/远距离场景建议开）")
    p.add_argument("--record", action="store_true", help="录制标注视频")
    p.add_argument("--log", action="store_true", help="角度写 CSV")
    p.add_argument("--outdir", default=os.path.join(HERE, "out"))
    p.add_argument("--headless", type=int, default=0,
                   help="无窗口模式：处理 N 帧后退出并存图（用于自动化验证）")
    p.add_argument("--image", default=None,
                   help="离线单图/目录模式：处理指定图片或目录，不打开摄像头")
    p.add_argument("--save-every", type=int, default=10,
                   help="headless 模式下每 N 帧存一张图")
    return p.parse_args()


def run_image_mode(a):
    """离线处理单张图片或整个目录（不打开摄像头，适合调试与批量测试）"""
    import glob
    os.makedirs(a.outdir, exist_ok=True)
    est = HandPoseEstimator(num_hands=a.num_hands, det_conf=a.det_conf,
                            pres_conf=a.pres_conf, track_conf=a.track_conf)

    if os.path.isdir(a.image):
        files = sorted(sum([glob.glob(os.path.join(a.image, e))
                            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))
    else:
        files = [a.image]
    if not files:
        print("[main] 未找到图片:", a.image)
        return

    print(f"[main] 离线模式：{len(files)} 张图")
    roi = parse_roi(a.roi)
    for fp in files:
        frame = imread_u(fp)
        if frame is None:
            print("  跳过（读不了）:", fp)
            continue
        H, W = frame.shape[:2]
        hands, used_roi = detect_hands(est, frame, roi=roi, auto_roi=a.auto_roi)

        primary = None
        for h in hands:
            lm = h.lm3d if np.any(h.lm3d) else h.lm2d
            st = compute_hand_state(lm, is_right=h.is_right)
            if primary is None:
                primary = (st, h)
            pts = draw_skeleton(frame, h.lm2d)
            draw_angle_labels(frame, pts, st)

        tag = os.path.splitext(os.path.basename(fp))[0]
        if primary is not None:
            st, h = primary
            draw_hud(frame, [f"{'Right' if st.is_right else 'Left'}  {gesture_hint_en(st)}"],
                     origin=(16, 26))
            if not a.no_bars:
                draw_curl_bars(frame, st, origin=(16, H - 6 * 26 - 12))
            if a.panel:
                pw, ph = 340, min(H, 660)
                panel = build_panel(st, 0.0, panel_w=pw, panel_h=ph)
                blend_panel(frame, panel, pos=(W - pw, 0), alpha=0.80)
            print(f"  {tag:<28} 手={len(hands)}  "
                  f"ROI={'auto' if used_roi else 'full'}  "
                  f"MCP/PIP/DIP(食指)={st.fingers[1].mcp:.0f}/{st.fingers[1].pip:.0f}/"
                  f"{st.fingers[1].dip:.0f}  弯曲={st.total_curl:.2f}  {gesture_hint(st)}")
        else:
            draw_hud(frame, ["no hand detected"], origin=(16, 26))
            print(f"  {tag:<28} 手=0")

        out = os.path.join(a.outdir, f"img_{tag}.jpg")
        imwrite_u(out, frame)
    est.close()
    print(f"[main] 结果 → {a.outdir}")


def main():
    a = parse_args()

    if a.image:
        run_image_mode(a)
        return

    os.makedirs(a.outdir, exist_ok=True)

    cam = Camera(index=a.camera, width=a.width, height=a.height, fps=a.fps,
                 mirror=not a.no_mirror)
    print("[main]", cam.info())
    est = HandPoseEstimator(num_hands=a.num_hands, det_conf=a.det_conf,
                            pres_conf=a.pres_conf, track_conf=a.track_conf)
    if not CNFont().ok:
        print("[main] ⚠️ 未找到中文字体，面板中文可能显示为方块（可改用 --no-panel）")

    writer = None
    logf = None
    csv_w = None
    roi = parse_roi(a.roi)
    t_start = time.perf_counter()
    t_prev = time.perf_counter()
    fps_avg = 0.0
    frame_id = 0
    paused = False
    recording = False
    save_count = 0

    if a.record and not a.headless:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(os.path.join(a.outdir, "annotated.mp4"),
                                 fourcc, a.fps, (cam.width, cam.height))
        recording = True
        print("[main] 录制中 →", os.path.join(a.outdir, "annotated.mp4"))

    try:
        while True:
            if not paused:
                ok, frame = cam.read()
                if not ok:
                    print("[main] 读帧失败，退出")
                    break
                frame_id += 1
            else:
                time.sleep(0.03)
                if frame is None:
                    continue

            t0 = time.perf_counter()

            # ---------------- 推理 ----------------
            ts_ms = int((time.perf_counter() - t_start) * 1000)
            # ⚠️ auto-roi 在"检不到手"时会多跑 4 次检测（5 倍开销），
            #    所以每 3 帧才试一次，避免掉了帧率。
            use_auto = a.auto_roi and (frame_id <= 5 or frame_id % 3 == 0)
            hands, used_roi = detect_hands(est, frame, roi=roi,
                                           auto_roi=use_auto, ts=ts_ms)

            # ---------------- 计算 + 绘制 ----------------
            primary = None
            for h in hands:
                # 用世界坐标(3D 米)算角度；没有则退化为 2D
                lm = h.lm3d if np.any(h.lm3d) else h.lm2d
                st = compute_hand_state(lm, is_right=h.is_right)
                if primary is None:
                    primary = (st, h)
                H, W = frame.shape[:2]
                pts = draw_skeleton(frame, h.lm2d)
                draw_angle_labels(frame, pts, st)

            # ---------------- 面板 + HUD ----------------
            if primary is not None:
                st, h = primary
                side = "Right" if st.is_right else "Left"
                draw_hud(frame, [f"{side}  {gesture_hint_en(st)}  curl {st.total_curl:.2f}"],
                         origin=(16, 26))
            else:
                draw_hud(frame, ["no hand detected"], origin=(16, 26))

            # FPS
            t_now = time.perf_counter()
            inst = 1.0 / max(1e-6, t_now - t_prev)
            t_prev = t_now
            fps_avg = inst if fps_avg == 0 else 0.9 * fps_avg + 0.1 * inst

            if not a.no_bars and primary is not None:
                bars_y = cam.height - 6 * 26 - 12
                draw_curl_bars(frame, primary[0], origin=(16, bars_y))

            if a.panel and primary is not None:
                pw = 340
                ph = min(cam.height, 660)
                panel = build_panel(primary[0], fps_avg,
                                    extra=[f"hand score {primary[1].score:.2f}"],
                                    panel_w=pw, panel_h=ph)
                blend_panel(frame, panel, pos=(cam.width - pw, 0), alpha=0.80)

            # FPS 数字（左上角，用 cv2 画数字更快）
            cv2.putText(frame, f"FPS {fps_avg:5.1f}", (16, cam.height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"FPS {fps_avg:5.1f}", (16, cam.height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1, cv2.LINE_AA)

            infer_ms = (time.perf_counter() - t0) * 1000.0

            # ---------------- 输出 ----------------
            if writer is not None and recording:
                writer.write(frame)

            if a.log and primary is not None and csv_w is None:
                cols = ["frame", "time", "side", "fps", "infer_ms"] + \
                       list(primary[0].flat().keys())
                logf = open(os.path.join(a.outdir, "angles.csv"), "w", newline="", encoding="utf-8")
                csv_w = csv.DictWriter(logf, fieldnames=cols, extrasaction="ignore")
                csv_w.writeheader()
                print("[main] 角度日志 →", os.path.join(a.outdir, "angles.csv"))
            if csv_w is not None and primary is not None and not paused:
                row = {"frame": frame_id, "time": f"{time.time():.3f}",
                       "side": "R" if primary[0].is_right else "L",
                       "fps": f"{fps_avg:.1f}", "infer_ms": f"{infer_ms:.1f}"}
                row.update(primary[0].flat())
                csv_w.writerow(row)

            # ---------------- 无窗口测试模式 ----------------
            if a.headless:
                if frame_id % max(1, a.save_every) == 0 or frame_id <= 3:
                    p = os.path.join(a.outdir, f"frame_{frame_id:05d}.jpg")
                    imwrite_u(p, frame)
                    save_count += 1
                    print(f"  [{frame_id:>4}] 手={len(hands)} "
                          f"推理={infer_ms:5.1f}ms FPS={fps_avg:5.1f} → {os.path.basename(p)}")
                if frame_id >= a.headless:
                    break
                continue

            # ---------------- 显示 ----------------
            cv2.imshow("Hand Joint Angles  (q=quit  s=shot  m=mirror  space=pause  r=rec)",
                       frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("s"):
                p = os.path.join(a.outdir, f"shot_{int(time.time())}.jpg")
                imwrite_u(p, frame)
                print("[main] 截图 →", p)
            elif k == ord("m"):
                cam.mirror = not cam.mirror
                print("[main] mirror =", cam.mirror)
            elif k == ord(" "):
                paused = not paused
                print("[main] paused =", paused)
            elif k == ord("r"):
                recording = not recording
                print("[main] recording =", recording)

    except KeyboardInterrupt:
        print("\n[main] 用户中断")
    finally:
        cam.release()
        est.close()
        if writer is not None:
            writer.release()
        if logf is not None:
            logf.close()
        if not a.headless:
            cv2.destroyAllWindows()
        print(f"[main] 结束。共处理 {frame_id} 帧，平均 {fps_avg:.1f} FPS"
              + (f"，存图 {save_count} 张 → {a.outdir}" if a.headless else ""))


if __name__ == "__main__":
    main()
