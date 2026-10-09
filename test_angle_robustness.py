"""test_angle_robustness.py — 分析各关节角的「抗噪能力」

原理：MediaPipe 的 3D 关键点有毫米级误差。给关键点加不同幅度的高斯噪声，
      重复计算关节角，看每个角的标准差 —— 标准差越大说明该角越不可靠。

用途：写论文/报告时说明「哪些角度可信、哪些参考使用」。
"""
import numpy as np

from joint_angle import compute_hand_state

rng = np.random.default_rng(0)


def make_hand():
    """一个合理的握持手型（21×3, 单位米）"""
    lm = np.zeros((21, 3), float)
    lm[0] = [0.00, 0.00, 0.00]
    # 四指弯曲的中等握持
    for base, x in {5: 0.030, 9: 0.055, 13: 0.078, 17: 0.098}.items():
        lm[base] = [x, 0.085, 0.000]
        lm[base + 1] = [x, 0.120, 0.018]
        lm[base + 2] = [x, 0.140, 0.045]
        lm[base + 3] = [x, 0.148, 0.072]
    lm[1] = [-0.030, 0.030, 0.0]
    lm[2] = [-0.058, 0.052, 0.010]
    lm[3] = [-0.080, 0.074, 0.028]
    lm[4] = [-0.095, 0.090, 0.048]
    return lm


base = make_hand()
names = ["拇指", "食指", "中指", "无名指", "小指"]

print("=== 关键点加噪后，各关节角的标准差（度）===")
print("   噪声幅度 = 每个关键点每个轴上的高斯 σ（米）")
print()
hdr = f"{'噪声σ(mm)':>9} | " + " | ".join(
    f"{n:<16}" for n in ["食指 三关节", "中指 三关节", "拇指 MCP/IP"])
print(hdr)
print("-" * len(hdr))

for sig_mm in (0.0, 1.0, 2.0, 3.0, 5.0):
    sig = sig_mm / 1000.0
    reps = 1 if sig == 0 else 300
    buf = {k: [] for k in ["imcp", "ipip", "idip", "mmcp", "mpip", "mdip", "tmcp", "tip"]}
    for _ in range(reps):
        lm = base + (rng.normal(0, sig, base.shape) if sig > 0 else 0)
        st = compute_hand_state(lm)
        buf["imcp"].append(st.fingers[1].mcp)
        buf["ipip"].append(st.fingers[1].pip)
        buf["idip"].append(st.fingers[1].dip)
        buf["mmcp"].append(st.fingers[2].mcp)
        buf["mpip"].append(st.fingers[2].pip)
        buf["mdip"].append(st.fingers[2].dip)
        buf["tmcp"].append(st.fingers[0].mcp)
        buf["tip"].append(st.fingers[0].pip)
    g = lambda k: np.nanstd(buf[k])
    print(f"{sig_mm:>9.1f} | "
          f"MCP {g('imcp'):4.1f} PIP {g('ipip'):4.1f} DIP {g('idip'):5.1f} | "
          f"MCP {g('mmcp'):4.1f} PIP {g('mpip'):4.1f} DIP {g('mdip'):5.1f} | "
          f"MCP {g('tmcp'):4.1f} IP {g('tip'):4.1f}")

print()
print("=== 相对可靠性（以 2mm 噪声为例，取最大者为 1）===")
sig = 0.002
buf = {k: [] for k in ["imcp", "ipip", "idip", "tmcp", "tip"]}
for _ in range(300):
    st = compute_hand_state(base + rng.normal(0, sig, base.shape))
    buf["imcp"].append(st.fingers[1].mcp)
    buf["ipip"].append(st.fingers[1].pip)
    buf["idip"].append(st.fingers[1].dip)
    buf["tmcp"].append(st.fingers[0].mcp)
    buf["tip"].append(st.fingers[0].pip)
sd = {k: np.nanstd(v) for k, v in buf.items()}
mx = max(sd.values())
for k, lbl in [("imcp", "食指 MCP"), ("ipip", "食指 PIP"), ("idip", "食指 DIP"),
               ("tmcp", "拇指 MCP"), ("tip", "拇指 IP")]:
    bar = "#" * max(1, int(30 * sd[k] / mx))
    verdict = ("可信" if sd[k] < 3 else ("参考" if sd[k] < 8 else "不可靠"))
    print(f"  {lbl:<10} σ={sd[k]:5.2f}°  {bar:<30} {verdict}")

print()
print("结论：")
print("  · DIP 角由【很短的末节】(dip->tip ≈ 1-2 cm) 决定 → 对关键点误差最敏感")
print("  · MCP / PIP 由较长骨段决定 → 抗噪好，是主要可用的指标")
print("  · 建议：报告时以 MCP + PIP 为主，DIP 仅作参考（或与时序平滑一起用）")
