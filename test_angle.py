"""_test_angle.py — 用合成关键点验证关节角公式是否正确

构造已知角度的"食指"，看 compute_hand_state 是否给出预期值。
（只改食指的 MCP/PIP/DIP，其余指头固定为伸直，避免互相覆盖索引）
"""
import math

import numpy as np

from joint_angle import compute_hand_state


def build_hand(pip_angle_deg: float, mcp_angle_deg: float = 180.0) -> np.ndarray:
    """构造 21 点手模：食指按指定 MCP/PIP 角弯曲，其余指伸直

    角度定义与代码一致：
      MCP 角 = ∠(腕, 食指MCP, 食指PIP)
      PIP 角 = ∠(食指MCP, 食指PIP, 食指DIP)
    """
    lm = np.zeros((21, 3), dtype=float)
    lm[0] = [0.0, 0.0, 0.0]                      # 腕

    # --- 食指 5,6,7,8 ---
    mcp = np.array([0.03, 0.09, 0.0])
    lm[5] = mcp
    u = mcp - lm[0]                     # 腕->MCP
    u = u / np.linalg.norm(u)
    # MCP 角 = ∠(腕,mcp,pip) = (mcp->腕) 与 (mcp->pip) 的夹角
    # (mcp->腕) = -u；把 -u 绕 z 转 mcp_angle 度即得 (mcp->pip)
    a1 = math.radians(mcp_angle_deg)
    t0 = np.array([-u[0], -u[1], 0.0])
    d1 = np.array([t0[0] * math.cos(a1) - t0[1] * math.sin(a1),
                   t0[0] * math.sin(a1) + t0[1] * math.cos(a1), 0.0])
    pip = mcp + d1 * 0.04
    lm[6] = pip
    # PIP 角 = ∠(mcp,pip,dip) = (pip->mcp) 与 (pip->dip) 的夹角
    # (pip->mcp) = -d1；把它绕 z 转 pip_angle 度即得 (pip->dip)
    t1 = np.array([-d1[0], -d1[1], 0.0])
    a2 = math.radians(pip_angle_deg)
    d2 = np.array([t1[0] * math.cos(a2) - t1[1] * math.sin(a2),
                   t1[0] * math.sin(a2) + t1[1] * math.cos(a2), 0.0])
    dip = pip + d2 * 0.03
    lm[7] = dip
    lm[8] = dip + d2 * 0.02                          # 伸直一段

    # --- 中指/无名/小指：完全伸直（沿 +y，指根 9/13/17）---
    for base, x in {9: 0.055, 13: 0.080, 17: 0.100}.items():
        lm[base] = [x, 0.085, 0.0]
        lm[base + 1] = [x, 0.130, 0.0]
        lm[base + 2] = [x, 0.160, 0.0]
        lm[base + 3] = [x, 0.180, 0.0]

    # --- 拇指 1,2,3,4：张开姿态 ---
    lm[1] = [-0.030, 0.030, 0.0]
    lm[2] = [-0.060, 0.055, 0.0]
    lm[3] = [-0.085, 0.075, 0.0]
    lm[4] = [-0.100, 0.090, 0.0]
    return lm


print("=== 合成测试：设定的食指 MCP/PIP 角 vs 计算值 ===")
print(f"{'设定MCP':>8} {'设定PIP':>8} | {'算出MCP':>8} {'算出PIP':>8} {'算出DIP':>8} {'curl':>6}")
ok = True
for mcp_set, pip_set in [(180, 180), (170, 150), (160, 120), (150, 90), (140, 60)]:
    f = compute_hand_state(build_hand(pip_set, mcp_set)).fingers[1]
    e_mcp, e_pip = abs(f.mcp - mcp_set), abs(f.pip - pip_set)
    if e_mcp >= 5 or e_pip >= 5:
        ok = False
    print(f"{mcp_set:>8} {pip_set:>8} | {f.mcp:>8.1f} {f.pip:>8.1f} {f.dip:>8.1f} {f.curl:>6.2f}"
          f"   误差 MCP {e_mcp:4.1f}° PIP {e_pip:4.1f}°")

print()
print("=== 交叉检验：用同一套关键点，pip 角应随设定单调变化 ===")
vals = [compute_hand_state(build_hand(p)).fingers[1].pip for p in (180, 150, 120, 90, 60)]
print("  设定 180/150/120/90/60 →", " / ".join(f"{v:.1f}" for v in vals))
print("  单调递减:", "OK" if all(vals[i] > vals[i+1] for i in range(4)) else "FAIL")

print()
print("总判定:", "✅ 角度公式正确" if ok else "❌ 公式或测试仍有问题")
