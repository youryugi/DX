# -*- coding: utf-8 -*-
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# -------- 参数 --------
GRAPHML_PATH = r"cached_network_34.59882_34.66824_135.48613_135.55117.graphml"
CSV_FILES = [
    r"edge_freq_home-school34.59882_34.66788_135.48613_135.55119.csv",
    r"edge_freq_home-station_34.59882-34.66824-135.48613-135.55117.csv",
    r"edge_freq_office-station_34.59882-34.66824-135.48639-135.55108.csv",
    r"edge_freq_school-station_34.59882-34.66824-135.48641-135.55119.csv",
    # ... 继续加
]
OUT_FIG_PATH = "./edge_frequency_sum.png"
CMAP_NAME = "viridis"
USE_LOG_SCALE = False
LINEWIDTH_MIN, LINEWIDTH_MAX = 0.5, 4.0
MISSING_FREQ_COLOR = (0, 0, 0, 0.15)

# -------- 1. 读图 --------
G = ox.load_graphml(GRAPHML_PATH)
edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()

# u/v 类型统一
def unify_uv(df):
    try:
        df["u"] = df["u"].astype(int)
        df["v"] = df["v"].astype(int)
    except Exception:
        df["u"] = df["u"].astype(str)
        df["v"] = df["v"].astype(str)
    return df

edges_gdf = unify_uv(edges_gdf)

# -------- 2. 读取多个 CSV 并求和 --------
sum_df = None
for path in CSV_FILES:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # 你前面代码里列名是 freq，这里也保持一致
    assert {"u", "v", "freq"} <= set(df.columns), f"{path} 缺少列 u,v,freq"
    df = unify_uv(df)[["u", "v", "freq"]]
    sum_df = df if sum_df is None else pd.concat([sum_df, df], axis=0)

# groupby 求每条边的总频率
sum_df = sum_df.groupby(["u", "v"], as_index=False)["freq"].sum()

# 合并到边
edges_gdf = edges_gdf.merge(sum_df, on=["u", "v"], how="left")
edges_gdf["freq"] = edges_gdf["freq"].fillna(0)

# -------- 3. 颜色和线宽 --------
freq_values = edges_gdf["freq"].values
if USE_LOG_SCALE:
    import numpy as np
    freq_plot_vals = np.log10(freq_values + 1e-9)
else:
    freq_plot_vals = freq_values

vmin, vmax = freq_plot_vals.min(), freq_plot_vals.max()
norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
cmap = plt.get_cmap(CMAP_NAME)

colors = [MISSING_FREQ_COLOR if f == 0 else cmap(norm(fv))
          for f, fv in zip(freq_values, freq_plot_vals)]

if vmax > vmin:
    widths = LINEWIDTH_MIN + (freq_plot_vals - vmin) / (vmax - vmin) * (LINEWIDTH_MAX - LINEWIDTH_MIN)
else:
    widths = [LINEWIDTH_MIN] * len(freq_plot_vals)

edges_gdf["color"] = colors
edges_gdf["linewidth"] = widths

# -------- 4. 绘图 --------
# ===================== 4. 颜色 & 线宽映射 =====================
freq_values = edges_gdf["freq"].values

# >>> 插入：截断+变换 <<<
import numpy as np
# 1) 分位截断，压低极端高值
P_LOW, P_HIGH = 1, 99
low, high = np.percentile(freq_values, [P_LOW, P_HIGH])
freq_clip = np.clip(freq_values, low, high)

# 2) 选择一种拉伸方式（挑一个用即可）
#freq_plot_vals = np.log10(freq_clip + 1e-9)    # 对数
#freq_plot_vals = np.sqrt(freq_clip)              # 开根号
freq_plot_vals = np.power(freq_clip, 0.3)      # 幂次
# <<< 结束 >>>

vmin, vmax = freq_plot_vals.min(), freq_plot_vals.max()
norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
cmap = plt.get_cmap(CMAP_NAME)

# 颜色数组
colors = []
for f, fv in zip(freq_values, freq_plot_vals):
    if f == 0:
        colors.append(MISSING_FREQ_COLOR)
    else:
        colors.append(cmap(norm(fv)))

# 线宽数组
if vmax > vmin:
    widths = LINEWIDTH_MIN + (freq_plot_vals - vmin) / (vmax - vmin) * (LINEWIDTH_MAX - LINEWIDTH_MIN)
else:
    widths = [LINEWIDTH_MIN] * len(freq_plot_vals)

fig, ax = plt.subplots(figsize=(10, 10))
edges_gdf.plot(color=edges_gdf["color"], linewidth=edges_gdf["linewidth"], ax=ax)
ax.set_axis_off()
ax.set_title("Edge Frequency (SUM of CSVs)", fontsize=16)

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Frequency (log scale)" if USE_LOG_SCALE else "Frequency", fontsize=12)

plt.tight_layout()
plt.savefig(OUT_FIG_PATH, dpi=300)
plt.show()
