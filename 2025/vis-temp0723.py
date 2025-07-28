# -*- coding: utf-8 -*-
"""
Plot edge frequencies on an existing road network (GraphML) using a CSV (u, v, frequency).

Requirements:
    pip install osmnx geopandas pandas matplotlib

Author: (your name)
"""

import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
# ===================== 1. 路径与参数设置 =====================
GRAPHML_PATH = r"cached_network_34.59882_34.66824_135.48613_135.55117.graphml"          # ← 你的graphml文件路径
FREQ_CSV_PATH = r"edge_freq_home-station_34.59882-34.66824-135.48613-135.55117.csv"           # ← 你的csv文件路径（u,v,frequency）
OUT_FIG_PATH = "./edge_frequency_map.png"    # 输出图路径
CMAP_NAME = "viridis"                        # 颜色映射，可换 "plasma","magma","inferno" 等
USE_LOG_SCALE = False                        # 若频率跨度很大，可改为 True 用对数
LINEWIDTH_MIN = 0.5                          # 最小线宽
LINEWIDTH_MAX = 4.0                          # 最大线宽
MISSING_FREQ_COLOR = (0, 0, 0, 0.15)         # 没有频率值的边（透明灰）

# ===================== 2. 读取数据 =====================
print("Loading graph...")
G = ox.load_graphml(GRAPHML_PATH)  # MultiDiGraph

print("Converting graph to GeoDataFrame...")
edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()  # columns: u, v, key, geometry, ...

print("Loading frequency CSV...")
freq_df = pd.read_csv(FREQ_CSV_PATH)

# 确保列名正确
freq_df.columns = [c.strip().lower() for c in freq_df.columns]
assert {"u", "v", "freq"} <= set(freq_df.columns), "CSV必须包含列: u, v, frequency"

# 类型对齐（GraphML里u/v通常是int或str，确保一致）
# 如果你的u/v是字符串，可以去掉astype(int)
try:
    edges_gdf["u"] = edges_gdf["u"].astype(int)
    edges_gdf["v"] = edges_gdf["v"].astype(int)
    freq_df["u"] = freq_df["u"].astype(int)
    freq_df["v"] = freq_df["v"].astype(int)
except Exception:
    # 如果不是纯数字ID就保持字符串
    edges_gdf["u"] = edges_gdf["u"].astype(str)
    edges_gdf["v"] = edges_gdf["v"].astype(str)
    freq_df["u"] = freq_df["u"].astype(str)
    freq_df["v"] = freq_df["v"].astype(str)

# ===================== 3. 合并频率到边 =====================
# 多重边的情况(key不同)：CSV只有u,v，先按(u,v) merge，再把同(u,v)的频率分配给所有key
edges_gdf = edges_gdf.merge(freq_df, on=["u", "v"], how="left")

# 缺失频率设为0
edges_gdf["freq"] = edges_gdf["freq"].fillna(0)

# ===================== 4. 颜色 & 线宽映射 =====================
freq_values = edges_gdf["freq"].values

if USE_LOG_SCALE:
    # 避免log(0)，加一个很小偏移
    import numpy as np
    freq_plot_vals = np.log10(freq_values + 1e-9)
else:
    freq_plot_vals = freq_values

vmin, vmax = freq_plot_vals.min(), freq_plot_vals.max()
norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
cmap = plt.get_cmap(CMAP_NAME)

# 颜色数组
colors = []
for f, fv in zip(freq_values, freq_plot_vals):
    if f == 0:  # 没频率或为0
        colors.append(MISSING_FREQ_COLOR)
    else:
        colors.append(cmap(norm(fv)))

# 线宽数组
if vmax > vmin:
    widths = LINEWIDTH_MIN + (freq_plot_vals - vmin) / (vmax - vmin) * (LINEWIDTH_MAX - LINEWIDTH_MIN)
else:
    widths = [LINEWIDTH_MIN] * len(freq_plot_vals)

edges_gdf["color"] = colors
edges_gdf["linewidth"] = widths

# ===================== 5. 绘图 =====================
fig, ax = plt.subplots(figsize=(10, 10))
edges_gdf.plot(color=edges_gdf["color"], linewidth=edges_gdf["linewidth"], ax=ax)

# 去掉坐标轴，或保留以便定位
ax.set_axis_off()
ax.set_title("Edge Frequency Visualization", fontsize=16)

# 添加颜色条
# 先创建一个ScalarMappable用来生成colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # 只用于colorbar
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Frequency (log scale)" if USE_LOG_SCALE else "Frequency", fontsize=12)

plt.tight_layout()
plt.savefig(OUT_FIG_PATH, dpi=300)
plt.show()

print(f"Done! Figure saved to: {OUT_FIG_PATH}")
