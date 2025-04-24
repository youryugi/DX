import pickle
from collections import defaultdict
from datetime import datetime
import os

# ========== 设置 ==========
input_file = "edge_shadow_ratios_20241205_0900_1000_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl"
tol = 0.01  # shadow_ratio 差值小于此视为稳定
# ==========================

# ========== 读取 shadow ratio 数据 ==========
with open(input_file, "rb") as f:
    precomputed = pickle.load(f)

print(f"[INFO] 已读取 shadow_ratio 数据，共 {len(precomputed):,} 条")

# ========== 整理为每条边的时间序列 ==========
edge_series = defaultdict(list)

for (u, v, k, t), ratio in precomputed.items():
    minute = t.hour * 60 + t.minute  # 使用分钟表示时间（方便）
    edge_series[(u, v, k)].append((minute, ratio))

# ========== 生成 trend ==========
edge_trend_map = {}

for key, series in edge_series.items():
    series.sort()  # 按时间升序排列
    segs = []

    start_minute, prev_ratio = series[0]
    current_type = None

    for i in range(1, len(series)):
        minute, ratio = series[i]
        delta = ratio - prev_ratio

        if abs(delta) < tol:
            new_type = "stable"
        elif delta > 0:
            new_type = "increasing"
        else:
            new_type = "decreasing"

        if current_type is None:
            current_type = new_type
        elif new_type != current_type:
            # 保存上一段
            segs.append({
                "start": start_minute,
                "end": series[i - 1][0],
                "type": current_type
            })
            start_minute = series[i - 1][0]
            current_type = new_type

        prev_ratio = ratio

    # 收尾最后一段
    segs.append({
        "start": start_minute,
        "end": series[-1][0],
        "type": current_type
    })

    edge_trend_map[key] = segs

print(f"[INFO] trend 提取完成，共 {len(edge_trend_map)} 条边")

# ========== 保存 ==========
output_file = input_file.replace("edge_shadow_ratios", "edge_trend_map")

with open(output_file, "wb") as f:
    pickle.dump(edge_trend_map, f)

print(f"[SUCCESS] trend 数据已保存至: {output_file}")
