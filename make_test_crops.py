"""_make_test_crops.py — 从 EgoTactile 广角帧里裁出"手部特写"，用于验证流水线

模拟你固定机位的真实取景（手占画面较大比例）。
自动搜索能检到手的裁剪窗口，存成 test_images2/。
"""
import glob
import os

import cv2

from hand_pose import HandPoseEstimator

SRC = sorted(glob.glob("test_images/*.jpg"))
OUT = "test_images2"
os.makedirs(OUT, exist_ok=True)

est = HandPoseEstimator(num_hands=2, det_conf=0.3, pres_conf=0.3, track_conf=0.3,
                        verbose=False)

# 搜索：不同中心与尺度（以画面中心为基准的裁剪比例）
WINDOWS = [
    (0.30, 0.25, 0.85, 0.95, "rc"),
    (0.15, 0.25, 0.70, 0.95, "lc"),
    (0.25, 0.35, 0.75, 1.00, "bc"),
    (0.35, 0.30, 0.95, 0.90, "rr"),
    (0.05, 0.30, 0.65, 0.90, "ll"),
    (0.25, 0.15, 0.75, 0.80, "tc"),
]

found = 0
for f in SRC:
    img = cv2.imread(f)
    H, W = img.shape[:2]
    name = os.path.splitext(os.path.basename(f))[0]
    for (x1f, y1f, x2f, y2f, tag) in WINDOWS:
        x1, y1, x2, y2 = int(x1f * W), int(y1f * H), int(x2f * W), int(y2f * H)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # 放大 2x 让手更大
        crop2 = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                           interpolation=cv2.INTER_CUBIC)
        hands = est.detect(crop2)
        if hands:
            p = os.path.join(OUT, f"{name}_{tag}.jpg")
            cv2.imwrite(p, crop2)
            print(f"  ✓ {name}_{tag}.jpg  检到 {len(hands)} 手  "
                  f"手别={'R' if hands[0].is_right else 'L'}  "
                  f"score={hands[0].score:.2f}")
            found += 1
            break

est.close()
print(f"\n共生成 {found} 张手部特写 → {OUT}/")
