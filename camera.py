"""camera.py — 摄像头采集封装

设计目的：把所有与「相机」相关的代码集中在这里。
以后换成 RealSense / Orbbec 时，**只需要改这个文件**，其他模块不用动。

支持：
  · 普通 USB / UVC 摄像头（OpenCV VideoCapture）
  · 预留 RealSense / Orbbec 接口（见文件末尾注释）
"""
from __future__ import annotations

import platform
import time

import cv2
import numpy as np

# Windows 上 DSHOW 通常比默认 MSMF 更稳、启动更快
_IS_WIN = platform.system() == "Windows"


class Camera:
    """统一的摄像头接口

    Usage:
        cam = Camera(index=0)
        ok, frame = cam.read()
        ...
        cam.release()
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, mirror: bool = True, prefer_dshow: bool | None = None):
        self.index = index
        self.mirror = mirror
        self.backend_name = "-"

        prefer_dshow = _IS_WIN if prefer_dshow is None else prefer_dshow
        backends = []
        if prefer_dshow:
            backends += [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")]
        else:
            backends += [(cv2.CAP_ANY, "ANY")]

        self.cap = None
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                self.cap = cap
                self.backend_name = name
                break
            cap.release()

        if self.cap is None:
            raise RuntimeError(
                f"打不开摄像头 index={index}。请确认：\n"
                f"  · 摄像头没有被其他程序占用（微信/QQ/浏览器/会议软件）\n"
                f"  · 换一个 index 试试（--camera 1 / 2 ...）\n"
                f"  · Windows 设置 → 隐私 → 相机，允许桌面应用访问"
            )

        # 请求目标参数（相机不一定全支持，读回实际值）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 降低延迟
        except Exception:
            pass

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_req = fps
        self.fps_actual = self.cap.get(cv2.CAP_PROP_FPS)

    # ---------------------------------------------------------------- read
    def read(self):
        """返回 (ok, frame)。frame 为 BGR；mirror=True 时已水平翻转"""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def info(self) -> str:
        return (f"camera#{self.index} [{self.backend_name}] "
                f"{self.width}x{self.height} @{self.fps_actual:.0f}fps "
                f"(请求 {self.fps_req})  mirror={'on' if self.mirror else 'off'}")

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# ---------------------------------------------------------------------------
# 以后换相机：只需要在这里加一个同名接口的类，main.py 里换一行即可
#
# class RealSenseCamera:
#     """Intel RealSense (pyrealsense2)"""
#     def __init__(self, width=1280, height=720, fps=30, mirror=False):
#         import pyrealsense2 as rs
#         self.pipe = rs.pipeline()
#         cfg = rs.config()
#         cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
#         self.pipe.start(cfg)
#         self.mirror = mirror
#     def read(self):
#         frames = self.pipe.wait_for_frames()
#         color = frames.get_color_frame()
#         if not color: return False, None
#         img = np.asanyarray(color.get_data())
#         return True, cv2.flip(img, 1) if self.mirror else img
#     def release(self): self.pipe.stop()
#
# class OrbbecCamera:
#     """Orbbec (pyorbbecsdk)"""
#     ... 同理：提供 read() / release() / info()
# ---------------------------------------------------------------------------
