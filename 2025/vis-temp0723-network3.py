# -*- coding: utf-8 -*-
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
# 顶部 import 里补一行
import geopandas as gpd

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
    assert {"u", "v", "freq"} <= set(df.columns), f"{path} 缺少列 u,v,freq"
    df = unify_uv(df)[["u", "v", "freq"]]
    sum_df = df if sum_df is None else pd.concat([sum_df, df], axis=0)

# groupby 求每条边的总频率
sum_df = sum_df.groupby(["u", "v"], as_index=False)["freq"].sum()

# 合并到边
edges_gdf = edges_gdf.merge(sum_df, on=["u", "v"], how="left")
edges_gdf["freq"] = edges_gdf["freq"].fillna(0)

# -------- 3. 颜色和线宽（原始版本）--------
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

# -------- 4. 绘图（你后面插入的截断+变换版本）--------
# ===================== 4. 颜色 & 线宽映射 =====================
freq_values = edges_gdf["freq"].values

# >>> 插入：截断+变换 <<<
import numpy as np
P_LOW, P_HIGH = 1, 99
low, high = np.percentile(freq_values, [P_LOW, P_HIGH])
freq_clip = np.clip(freq_values, low, high)

#freq_plot_vals = np.log10(freq_clip + 1e-9)    # 对数
#freq_plot_vals = np.sqrt(freq_clip)            # 开根号
freq_plot_vals = np.power(freq_clip, 0.3)       # 幂次
# <<< 结束 >>>

vmin, vmax = freq_plot_vals.min(), freq_plot_vals.max()
norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
cmap = plt.get_cmap(CMAP_NAME)

colors = []
for f, fv in zip(freq_values, freq_plot_vals):
    if f == 0:
        colors.append(MISSING_FREQ_COLOR)
    else:
        colors.append(cmap(norm(fv)))

if vmax > vmin:
    widths = LINEWIDTH_MIN + (freq_plot_vals - vmin) / (vmax - vmin) * (LINEWIDTH_MAX - LINEWIDTH_MIN)
else:
    widths = [LINEWIDTH_MIN] * len(freq_plot_vals)

edges_gdf["color"] = colors
edges_gdf["linewidth"] = widths

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

# ===================== 5. 高频子网 + Steiner Tree 连接 =====================
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import numpy as np
import pandas as pd

# 5.1 高频边筛选
Q = 90  # 取前 10% 高频
thr = np.percentile(edges_gdf['freq'], Q)
hf_edges_gdf = edges_gdf[edges_gdf['freq'] >= thr].copy()
print(f"阈值 {thr:.1f}，高频边 {len(hf_edges_gdf)} 条")

# 5.2 terminals（先统一类型，再过滤不存在的）
# 用图里任意节点推断类型
sample_node = next(iter(G.nodes()))
node_type = type(sample_node)

hf_edges_gdf['u'] = hf_edges_gdf['u'].astype(node_type)
hf_edges_gdf['v'] = hf_edges_gdf['v'].astype(node_type)

terminals_raw = set(hf_edges_gdf['u']).union(hf_edges_gdf['v'])

# 5.3 把原始图转无向 + 简化成简单图（去掉多重边），避免 MultiGraph 的坑
G_u_multi = G.to_undirected()

# 把多重边压成单边：保留最短 weight（如果有 length）
WEIGHT_ATTR = 'length' if any('length' in d for _,_,d in G_u_multi.edges(data=True)) else None

G_simple = nx.Graph()
for u, v, data in G_u_multi.edges(data=True):
    w = data.get(WEIGHT_ATTR, 1)
    if G_simple.has_edge(u, v):
        # 只保留最小权重
        if w < G_simple[u][v].get(WEIGHT_ATTR, w):
            G_simple[u][v][WEIGHT_ATTR] = w
    else:
        if WEIGHT_ATTR:
            G_simple.add_edge(u, v, **{WEIGHT_ATTR: w})
        else:
            G_simple.add_edge(u, v)

# 过滤 terminals（必须在简化后的图里）
nodes_set = set(G_simple.nodes())
missing = [n for n in terminals_raw if n not in nodes_set]
if missing:
    print(f"[WARN] {len(missing)} terminals 不在图中，已丢弃。例：{missing[:5]}")
terminals = list(terminals_raw & nodes_set)

# 如果终端太少/已连通就直接跳过
if len(terminals) <= 1:
    print("终端数量 <=1，跳过 Steiner Tree。")
    T_simple = G_simple.subgraph(terminals).copy()
else:
    # 5.4 计算 Steiner Tree
    T_simple = steiner_tree(G_simple, terminals, weight=WEIGHT_ATTR)
    print(f"Steiner Tree: {T_simple.number_of_nodes()} nodes, {T_simple.number_of_edges()} edges")

# 5.5 把 Steiner Tree 边转为 GeoDataFrame，并补属性（geometry/freq/color/linewidth）
def sort_uv_row(row):
    return tuple(sorted((row['u'], row['v'])))

t_edges_df = pd.DataFrame(T_simple.edges(), columns=['u', 'v'])
t_edges_df['uv_sorted'] = t_edges_df.apply(sort_uv_row, axis=1)

edges_gdf_tmp = edges_gdf.copy()
edges_gdf_tmp['uv_sorted'] = edges_gdf_tmp.apply(lambda r: tuple(sorted((r['u'], r['v']))), axis=1)

t_edges_full = t_edges_df.merge(
    edges_gdf_tmp[['uv_sorted', 'u', 'v', 'key', 'geometry', 'freq', 'color', 'linewidth']],
    on='uv_sorted', how='left'
)
# 5.5 之后，生成 t_edges_full 后立刻转成 GeoDataFrame
t_edges_full = gpd.GeoDataFrame(t_edges_full, geometry='geometry', crs=edges_gdf.crs)

# 补默认值（保持你现有逻辑）
t_edges_full['freq'] = t_edges_full['freq'].fillna(1)
default_color = (0.2, 0.2, 0.2, 0.8)
t_edges_full['color'] = t_edges_full['color'].astype(object).apply(
    lambda x: default_color if pd.isna(x) else x
)
t_edges_full['linewidth'] = t_edges_full['linewidth'].fillna(1.5)

# 删掉 geometry 为空的行
t_edges_full = t_edges_full[~t_edges_full['geometry'].isna()]

# ---- 5.5 merge 之后立刻规范列名 ----
# 你原来的 merge
t_edges_full = t_edges_df.merge(
    edges_gdf_tmp[['uv_sorted', 'u', 'v', 'key', 'geometry', 'freq', 'color', 'linewidth']],
    on='uv_sorted', how='left', suffixes=('_tree', '')
)

# 如果 merge 产生了 u_tree/v_tree（或者 u_x/v_x），把想用的列统一成 u/v
rename_map = {}
for cand, target in [('u_tree', 'u'), ('v_tree', 'v'),
                     ('u_x', 'u'),    ('v_x', 'v'),
                     ('u_y', 'u'),    ('v_y', 'v')]:
    if cand in t_edges_full.columns and target not in t_edges_full.columns:
        rename_map[cand] = target
t_edges_full = t_edges_full.rename(columns=rename_map)

# 还可能同时存在 u 和 u_tree 两份，保留 edges_gdf 的那份，删掉多余的
for dup in ['u_tree', 'v_tree', 'u_x', 'v_x', 'u_y', 'v_y']:
    if dup in t_edges_full.columns and dup not in ['u', 'v']:
        t_edges_full = t_edges_full.drop(columns=[dup])

# 5.6 画图
# 5.6 画图时，用列表传参，确保走 geopandas 的 plot
fig2, ax2 = plt.subplots(figsize=(10, 10))
edges_gdf.plot(ax=ax2, color=(0, 0, 0, 0.08), linewidth=0.3)
hf_edges_gdf.plot(ax=ax2,
                  color=list(hf_edges_gdf['color']),
                  linewidth=list(hf_edges_gdf['linewidth']))
t_edges_full.plot(ax=ax2,
                  color=list(t_edges_full['color']),
                  linewidth=list(t_edges_full['linewidth']))
ax2.set_axis_off()
ax2.set_title("High-Frequency Network + Steiner Backbone", fontsize=16)
plt.tight_layout()
plt.savefig("edge_frequency_steiner.png", dpi=300)
plt.show()

# ===================== 6. 剪枝得到“少分叉整体路网” =====================
import networkx as nx
import geopandas as gpd
import numpy as np

# 6.0 先把 “高频边” + “Steiner边” 合并成一个 GeoDataFrame
backbone_edges = pd.concat([hf_edges_gdf[['u','v','key','geometry','freq','color','linewidth']],
                            t_edges_full[['u','v','key','geometry','freq','color','linewidth']]],
                           ignore_index=True).drop_duplicates(subset=['u','v','key'])
backbone_edges = gpd.GeoDataFrame(backbone_edges, geometry='geometry', crs=edges_gdf.crs)

# 6.1 构建无向简单图，方便剪枝
G_bb = nx.Graph()
for i, r in backbone_edges.iterrows():
    G_bb.add_edge(r['u'], r['v'],
                  idx=i,
                  freq=r['freq'],
                  length=r['geometry'].length)

# 6.2 迭代剪枝函数：删掉度=1的叶子边，但必须满足"弱枝"条件
def prune_leaves(G, freq_thr=0, len_thr=0):
    """
    从图中删除满足条件的叶子边。
    freq_thr: 叶子边的 freq 小于该值才删
    len_thr : 叶子边的 length 小于该值才删
    返回是否有删
    """
    to_remove = []
    for n in list(G.nodes()):
        if G.degree(n) == 1:
            # 只有一个邻居，找到那条边
            nbr = next(iter(G.neighbors(n)))
            data = G.get_edge_data(n, nbr)
            # Graph 里可能多个平行边被压成一个，这里只有一个 dict
            f = data['freq']
            L = data['length']
            if f <= freq_thr or L <= len_thr:
                to_remove.append((n, nbr))
    if not to_remove:
        return False
    G.remove_edges_from(to_remove)
    # 同时删掉孤立点
    iso = list(nx.isolates(G))
    if iso:
        G.remove_nodes_from(iso)
    return True

# 6.3 设置剪枝阈值（自行调参）
# 用分位数当阈值会更稳：比如删掉 freq 在后 10% 的叶子，同时长度 < 20m
leaf_freq_thr = np.percentile(backbone_edges['freq'], 10)
leaf_len_thr  = 20   # 单位同你的坐标系（一般是米）

# 6.4 迭代剪，直到不能再剪
for _ in range(50):            # 最多迭代 50 次
    changed = prune_leaves(G_bb, freq_thr=leaf_freq_thr, len_thr=leaf_len_thr)
    if not changed:
        break

# （可选）6.5 做个 k-core（k=2），去除度<2的小块
G_bb = nx.k_core(G_bb, k=2)

# 6.6 从剪枝后的图里取回边（通过 idx）
kept_idx = []
for u, v, data in G_bb.edges(data=True):
    kept_idx.append(data['idx'])

bb_pruned = backbone_edges.loc[kept_idx].copy()

# 6.7 可视化：背景灰 + 剪枝后的整体主干
fig4, ax4 = plt.subplots(figsize=(10,10))
edges_gdf.plot(ax=ax4, color=(0,0,0,0.08), linewidth=0.3)
bb_pruned.plot(ax=ax4,
               color=list(bb_pruned['color']),
               linewidth=list(bb_pruned['linewidth']))
ax4.set_axis_off()
ax4.set_title("Pruned Global Backbone (few branches)", fontsize=16)
plt.tight_layout()
plt.savefig("edge_frequency_backbone_pruned.png", dpi=300)
plt.show()
