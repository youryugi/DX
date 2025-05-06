"""
offline_build_node_minute_weight.py
-----------------------------------
一次性离线生成:
 1) flat_trend_lookup.pkl   # (u,v,k, minute) → weight
 2) node_minute_weight.pkl  # (node, minute)  → avg_weight
"""

import pickle
import pandas as pd
import os
from tqdm import tqdm
import time

# ===========================================================================
# 参数设置
# ===========================================================================
EDGE_TREND_PKL   = "edge_trend_map_20241205_0900_1000_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl"      # 你的 (u,v,k) -> [{'start', 'end', 'type'}]
GRAPH_PKL        = "osmnx_graph_34.65015207474573_34.62460168764661_135.5501897775654_135.51224918398333.pkl"         # 存有 networkx 图 G
FLAT_TREND_OUT    = "flat_trend_lookup.pkl"     # 输出：展平后的 trend 查表
NODE_WEIGHT_OUT   = "node_minute_weight.pkl"    # 输出：每个 node 每分钟平均权重


# ===============================================
# 趋势类型对应的权重（可以调整）
# ===============================================
TREND_W = {
    "increasing": 0.8,
    "stable":     1.0,
    "decreasing": 1.2
}

# ===============================================
# 读取 edge_trend_map.pkl
# ===============================================
print("📂 正在读取 edge_trend_map.pkl …")
with open(EDGE_TREND_PKL, "rb") as f:
    edge_trend_map = pickle.load(f)
print(f"✅ 加载完成，边数：{len(edge_trend_map):,}")

# ===============================================
# 将 edge_trend_map 转为 trend_interval_map 格式
# ===============================================
print("🔄 构建 trend_interval_map …")
trend_interval_map = {}
for key, seg_list in edge_trend_map.items():
    starts = [s["start"] for s in seg_list]
    ends   = [s["end"]   for s in seg_list]
    types  = [s["type"]  for s in seg_list]
    ivx = pd.IntervalIndex.from_arrays(starts, ends, closed="left")
    trend_interval_map[key] = (ivx, types)

# ===============================================
# A. 构建展平 trend 查表：(u,v,k,minute) → 权重
# ===============================================
print("⚙️ 正在构建 flat_trend_lookup …")
flat_trend_lookup = {}
for (u, v, k), (intervals, types) in tqdm(trend_interval_map.items(), desc="Flatten trend"):
    for idx, iv in enumerate(intervals):
        weight = TREND_W.get(types[idx], 1.0)
        for minute in range(iv.left, iv.right):
            flat_trend_lookup[(u, v, k, minute % 1440)] = weight

print(f"✅ flat_trend_lookup 构建完成，共 {len(flat_trend_lookup):,} 条")

# 保存 flat_trend_lookup.pkl
with open(FLAT_TREND_OUT, "wb") as f:
    pickle.dump(flat_trend_lookup, f)
print(f"📦 已保存 {FLAT_TREND_OUT}（{os.path.getsize(FLAT_TREND_OUT)/1e6:.1f} MB）")

# ===============================================
# B. 构建 node-minute 权重：(node, minute) → avg_weight
# ===============================================
print("📂 正在读取 osmnx_graph.pkl …")
with open(GRAPH_PKL, "rb") as f:
    G = pickle.load(f)

print(f"✅ 图加载完成：节点数 {len(G.nodes):,}，边数 {len(G.edges):,}")

print("⚙️ 正在构建 node_minute_weight …")
node_minute_weight = {}
for n in tqdm(G.nodes, desc="Processing nodes"):
    out_edges = [(n, nbr, k) for nbr in G[n] for k in G[n][nbr]]
    if not out_edges:
        for m in range(1440):
            node_minute_weight[(n, m)] = 1.0
        continue

    for m in range(1440):
        weights = [flat_trend_lookup.get((u, v, k, m), 1.0) for (u, v, k) in out_edges]
        node_minute_weight[(n, m)] = sum(weights) / len(weights)

print(f"✅ node_minute_weight 构建完成，共 {len(node_minute_weight):,} 条")

# 保存 node_minute_weight.pkl
with open(NODE_WEIGHT_OUT, "wb") as f:
    pickle.dump(node_minute_weight, f)
print(f"📦 已保存 {NODE_WEIGHT_OUT}（{os.path.getsize(NODE_WEIGHT_OUT)/1e6:.1f} MB）")

print("\n🎉 所有预计算完成！")
