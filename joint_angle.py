"""joint_angle.py — 手部关节角度计算

输入：MediaPipe 的 21 个关键点（世界坐标 3D 为主，2D 用于绘制）
输出：每根手指的 MCP / PIP / DIP 关节角 + 张开度 + 拇指-食指对合 + 各指伸展率

坐标约定（MediaPipe 21 点索引）：
    0      腕 wrist
    1-4    拇指 thumb:  cmc(1), mcp(2), ip(3),  tip(4)
    5-8    食指 index:  mcp(5), pip(6), dip(7), tip(8)
    9-12   中指 middle: mcp(9), pip(10), dip(11), tip(12)
    13-16  无名 ring:   mcp(13), pip(14), dip(15), tip(16)
    17-20  小指 pinky:  mcp(17), pip(18), dip(19), tip(20)

⚠️ 注意：拇指是 (cmc, mcp, ip, tip)，其余四指是 (mcp, pip, dip, tip) —— 语义不同！
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- 索引表
WRIST = 0
THUMB = (1, 2, 3, 4)        # cmc, mcp, ip,  tip
INDEX = (5, 6, 7, 8)        # mcp, pip, dip, tip
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

# (中文名, 英文名, 关节索引, 是否拇指)
FINGER_DEF = [
    ("拇指", "Thumb", THUMB, True),
    ("食指", "Index", INDEX, False),
    ("中指", "Middle", MIDDLE, False),
    ("无名指", "Ring", RING, False),
    ("小指", "Pinky", PINKY, False),
]
FINGER_EN = [d[1] for d in FINGER_DEF]

# 骨架连线（mediapipe 官方 HAND_CONNECTIONS）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                     # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),                     # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),                # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),              # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),             # 小指
    (0, 17),                                            # 掌根闭合
]
TIPS = [4, 8, 12, 16, 20]
MCPS = [2, 5, 9, 13, 17]


# ---------------------------------------------------------------- 基础工具
def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """关节 b 处的夹角 ∠(a-b-c)，返回角度（0~180）"""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cosv = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return math.degrees(math.acos(cosv))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------- 结果容器
@dataclass
class FingerAngles:
    name_cn: str
    name_en: str
    mcp: float = float("nan")        # 掌指关节
    pip: float = float("nan")        # 近端指间关节（拇指为 IP）
    dip: float = float("nan")        # 远端指间关节（拇指无 → NaN）
    extension: float = float("nan")  # 伸展率 |tip-wrist| / 掌长
    curl: float = 0.0                # 弯曲度 0(伸直)~1(握紧)

    def as_dict(self) -> dict:
        return {"mcp": self.mcp, "pip": self.pip, "dip": self.dip,
                "extension": self.extension, "curl": self.curl}


@dataclass
class HandState:
    is_right: bool = True
    fingers: list[FingerAngles] = field(default_factory=list)
    spread: dict = field(default_factory=dict)   # 相邻指张开角（度）
    opposition: float = float("nan")             # 拇指尖-食指尖距离 / 掌长
    palm_length: float = float("nan")            # 腕→中指MCP
    palm_width: float = float("nan")             # 食指MCP→小指MCP
    total_curl: float = 0.0                      # 四指（不含拇指）平均弯曲度

    def flat(self, prefix: str = "") -> dict:
        """展平成 {名称: 数值}，便于写 CSV"""
        d = {}
        for f in self.fingers:
            d[f"{prefix}{f.name_en}_mcp"] = round(f.mcp, 2)
            d[f"{prefix}{f.name_en}_pip"] = round(f.pip, 2)
            d[f"{prefix}{f.name_en}_dip"] = round(f.dip, 2)
            d[f"{prefix}{f.name_en}_curl"] = round(f.curl, 3)
            d[f"{prefix}{f.name_en}_ext"] = round(f.extension, 3)
        for k, v in self.spread.items():
            d[f"{prefix}spread_{k}"] = round(v, 2)
        d[f"{prefix}opposition"] = round(self.opposition, 3)
        d[f"{prefix}total_curl"] = round(self.total_curl, 3)
        return d


# ---------------------------------------------------------------- 主函数
def compute_hand_state(lm: np.ndarray, is_right: bool = True,
                       curl_lo: float = 180.0, curl_hi: float = 60.0) -> HandState:
    """由 21×3 关键点计算手部状态

    lm : (21,3) 关键点。推荐传**世界坐标(米)**；传归一化 2D 也能算，但尺度无意义
    curl_lo, curl_hi : 参考角 → 弯曲度 的映射区间
        参考角 = 180°（完全伸直）→ curl = 0
        参考角 =  60°（明显弯曲）→ curl = 1
    """
    st = HandState(is_right=is_right)
    wrist = lm[WRIST]

    st.palm_length = dist(lm[MIDDLE[0]], wrist)
    st.palm_width = dist(lm[INDEX[0]], lm[PINKY[0]])
    ref = st.palm_length if st.palm_length > 1e-6 else 1.0

    for name_cn, name_en, idx, is_thumb in FINGER_DEF:
        fa = FingerAngles(name_cn=name_cn, name_en=name_en)

        if is_thumb:
            cmc, mcp, ip, tip = idx          # 1,2,3,4
            fa.mcp = angle_at(lm[cmc], lm[mcp], lm[ip])    # ∠(cmc-mcp-ip)
            fa.pip = angle_at(lm[mcp], lm[ip], lm[tip])    # IP 角 ∠(mcp-ip-tip)
            fa.dip = float("nan")
            ref_angle = fa.pip
        else:
            mcp, pip, dip, tip = idx         # 5,6,7,8
            fa.mcp = angle_at(wrist, lm[mcp], lm[pip])     # ∠(腕-mcp-pip)
            fa.pip = angle_at(lm[mcp], lm[pip], lm[dip])   # ∠(mcp-pip-dip)
            fa.dip = angle_at(lm[pip], lm[dip], lm[tip])   # ∠(pip-dip-tip)
            ref_angle = fa.pip

        fa.extension = dist(lm[tip], wrist) / ref
        if not math.isnan(ref_angle):
            t = (curl_lo - ref_angle) / max(1e-6, (curl_lo - curl_hi))
            fa.curl = float(np.clip(t, 0.0, 1.0))
        st.fingers.append(fa)

    # 相邻手指张开角（指根相对腕的方向夹角）
    for k, (i1, i2) in enumerate([(INDEX[0], MIDDLE[0]),
                                  (MIDDLE[0], RING[0]),
                                  (RING[0], PINKY[0])]):
        nm = ["index-middle", "middle-ring", "ring-pinky"][k]
        st.spread[nm] = angle_at(lm[i1], wrist, lm[i2])
    st.spread["thumb-index"] = angle_at(lm[THUMB[0]], wrist, lm[INDEX[0]])

    # 拇指-食指指尖对合（归一化距离）
    st.opposition = dist(lm[THUMB[3]], lm[INDEX[3]]) / ref

    # 四指（不含拇指）平均弯曲度
    fc = [f.curl for f in st.fingers[1:]]
    st.total_curl = float(np.mean(fc)) if fc else 0.0
    return st


def gesture_hint(st: HandState) -> str:
    """粗略手势提示（仅显示用，不做严格分类）"""
    c = st.total_curl
    thumb_c = st.fingers[0].curl
    opp = st.opposition
    if c > 0.75:
        return "握拳 / 抓握"
    if c < 0.25 and st.spread.get("index-middle", 0) > 12:
        return "张开 / 伸展"
    if opp < 0.6 and c < 0.5:
        return "捏合 / 对指"
    if c < 0.5 and thumb_c > 0.4:
        return "半握"
    return "中间状态"
