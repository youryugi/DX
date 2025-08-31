# -*- coding: utf-8 -*-
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np

# ===== 参数 =====
GRAPHML_PATH = r"cached_network_34.59882_34.66824_135.48613_135.55117.graphml"
CSV_FILES = [
    r"edge_freq_home-school34.59882_34.66788_135.48613_135.55119.csv",
    r"edge_freq_home-station_34.59882-34.66824-135.48613-135.55117.csv",
    r"edge_freq_office-station_34.59882-34.66824-135.48639-135.55108.csv",
    r"edge_freq_school-station_34.59882-34.66824-135.48641-135.55119.csv",
    # ... 继续加
]
OUT_FIG_PATH = "./edge_frequency_highfreq_with_bg.png"

# 高频选择模式： 'quantile' 或 'topk'
SELECT_MODE = 'quantile'   # 'quantile' | 'topk'
Q_HIGH = 95                # 分位阈值（仅在 quantile 模式下使用）
TOP_K = 800                # 仅在 topk 模式下使用：显示前 K 条（按 freq 降序）

# 线宽（红色高频）
LINEWIDTH_MIN, LINEWIDTH_MAX = 0.6, 4.0
RED_COLOR = (0.85, 0.0, 0.0, 1.0)

# 背景路网（淡灰细线）
SHOW_BACKGROUND = True
BG_COLOR = (0, 0, 0, 0.08)
BG_LINEWIDTH = 0.25

# 分位截断（仅影响线宽映射，不影响是否入选高频）
P_LOW, P_HIGH = 1, 99
USE_TRANSFORM = "power"  # "power" | "sqrt" | "log"
USE_LOG_SCALE = False    # True 时等价于对数拉伸


# ===== 1) 读图 =====
G = ox.load_graphml(GRAPHML_PATH)
edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()

def unify_uv(df):
    try:
        df["u"] = df["u"].astype(int)
        df["v"] = df["v"].astype(int)
    except Exception:
        df["u"] = df["u"].astype(str)
        df["v"] = df["v"].astype(str)
    return df

edges_gdf = unify_uv(edges_gdf)

# ===== 2) 读 CSV 并求和 =====
sum_df = None
for path in CSV_FILES:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"u", "v", "freq"} <= set(df.columns), f"{path} 缺少列 u,v,freq"
    df = unify_uv(df)[["u", "v", "freq"]]
    sum_df = df if sum_df is None else pd.concat([sum_df, df], axis=0)

sum_df = sum_df.groupby(["u", "v"], as_index=False)["freq"].sum()
edges_gdf = edges_gdf.merge(sum_df, on=["u", "v"], how="left")
edges_gdf["freq"] = edges_gdf["freq"].fillna(0)

# ===== 3) 选择高频子集 =====
edges_gdf = edges_gdf.sort_values("freq", ascending=False)
freq_all = edges_gdf["freq"].values.astype(float)

if SELECT_MODE == 'topk':
    k = min(TOP_K, len(edges_gdf))
    hf_edges_gdf = edges_gdf.head(k).copy()
    thr_info = f"Top-{k} edges"
else:
    thr = np.percentile(freq_all, Q_HIGH) if len(freq_all) else 0.0
    hf_edges_gdf = edges_gdf[edges_gdf["freq"] >= thr].copy()
    # 兜底：若空，则降到 80 分位
    if hf_edges_gdf.empty and len(freq_all) > 0:
        thr = np.percentile(freq_all, 80.0)
        hf_edges_gdf = edges_gdf[edges_gdf["freq"] >= thr].copy()
        thr_info = f"≥80th percentile (fallback)"
    else:
        thr_info = f"≥{Q_HIGH}th percentile"

print(f"[Info] Selected {len(hf_edges_gdf)} high-frequency edges ({thr_info}).")

# 若仍为空，直接退出绘图避免空图
if hf_edges_gdf.empty:
    raise RuntimeError("高频筛选结果为空。请降低 Q_HIGH 或改用 SELECT_MODE='topk'。")

# ===== 4) 线宽映射（仅对 hf 子集）=====
freq = hf_edges_gdf["freq"].values.astype(float)

# 分位截断 -> 非线性拉伸
low, high = (np.percentile(freq, [P_LOW, P_HIGH]) if len(freq) else (0.0, 1.0))
freq_clip = np.clip(freq, low, high)

if USE_LOG_SCALE or USE_TRANSFORM == "log":
    plot_vals = np.log10(freq_clip + 1e-9)
elif USE_TRANSFORM == "sqrt":
    plot_vals = np.sqrt(freq_clip)
elif USE_TRANSFORM == "power":
    plot_vals = np.power(freq_clip, 0.3)
else:
    plot_vals = freq_clip

vmin, vmax = plot_vals.min(), plot_vals.max()
normed = (plot_vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(plot_vals)

linewidths = LINEWIDTH_MIN + normed * (LINEWIDTH_MAX - LINEWIDTH_MIN)
hf_edges_gdf["linewidth"] = linewidths
hf_edges_gdf["color"] = [RED_COLOR] * len(hf_edges_gdf)

# ===== 5) 绘图 =====
fig, ax = plt.subplots(figsize=(10, 10))

# 背景路网（可选）
if SHOW_BACKGROUND:
    edges_gdf.plot(color=BG_COLOR, linewidth=BG_LINEWIDTH, ax=ax)

# 高频子网（红色线宽编码频率）
hf_edges_gdf.plot(
    color=list(hf_edges_gdf["color"]),
    linewidth=list(hf_edges_gdf["linewidth"]),
    ax=ax
)

ax.set_axis_off()
ax.set_title(f"High-frequency Edges ({thr_info})", fontsize=16)

plt.tight_layout()
plt.savefig(OUT_FIG_PATH, dpi=300)
plt.show()
