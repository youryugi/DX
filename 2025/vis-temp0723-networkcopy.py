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
# ========= 5. 抽取高频子网络 =========
import networkx as nx
import numpy as np

# 1) 用分位数阈值
Q = 80                      # 取前 10% 高频边
thr = np.percentile(edges_gdf['freq'], Q)

hf_edges_gdf = edges_gdf[edges_gdf['freq'] >= thr].copy()
print(f"≥{Q}分位阈值 = {thr:.1f}，共 {len(hf_edges_gdf)} 条边")

# 2) 如果原始图是多边（MultiDiGraph），需要带 key
# 先保证 edges_gdf 里有 'key'
if 'key' not in hf_edges_gdf.columns:
    # osmnx.graph_to_gdfs 默认会给 key；若没有，可从 G 中取
    edges_all = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    hf_edges_gdf = hf_edges_gdf.merge(edges_all[['u','v','key']], on=['u','v'], how='left')

# 3) 用 edge_subgraph 抽取
edge_keys = list(zip(hf_edges_gdf['u'], hf_edges_gdf['v'], hf_edges_gdf['key']))
G_hf = G.edge_subgraph(edge_keys).copy()

# ========= 6. 可视化高频子网 =========
fig2, ax2 = plt.subplots(figsize=(10, 10))

# 背景：原始路网淡灰
edges_gdf.plot(color=(0,0,0,0.1), linewidth=0.3, ax=ax2)

# 高频子网：线宽 & 颜色（延用前面算好的）
hf_edges_gdf.plot(color=hf_edges_gdf['color'], linewidth=hf_edges_gdf['linewidth'], ax=ax2)

ax2.set_axis_off()
ax2.set_title(f"High-frequency Backbone (freq ≥ {thr:.1f}, top {100-Q}%)", fontsize=16)

plt.tight_layout()
plt.savefig("edge_frequency_backbone.png", dpi=300)
plt.show()

# ========= 7. 超简主干走廊提取（MST + 叶子修剪） =========
import numpy as np
import networkx as nx
from shapely.geometry import LineString

# -------- 参数（按需调）--------
Q = 80                 # 只把频率进入前 (100-Q)% 的边作为候选（例如 80 -> 取前20% 高频）
TARGET_TOTAL_KM = 20   # 目标：最终主干总长度（公里），可改为 10/15/30 等
LEAF_WEIGHT_Q = 0.40   # 修剪阈：小于候选集中第 40% 分位的叶子边会优先被剪
MAX_ITERS = 100000     # 安全上限，避免死循环

# -------- 1) 候选子网：高频边 --------
thr = np.percentile(edges_gdf['freq'], Q)
cand = edges_gdf[edges_gdf['freq'] >= thr].copy()
if 'length' not in cand.columns or cand['length'].isna().all():
    # 若没有 length，就用几何长度（需要米坐标；OSMnx图通常已有 length）
    # 这里兜底用几何的度量长度（近似），更严谨可先投影到米制坐标系再算
    cand['length'] = cand.geometry.length

# -------- 2) 构建简单无向图（合并多重边：选权重大的）--------
# 权重 = 频率 * 长度（让“又常用又长”的段优先进入骨架）
cand['weight'] = cand['freq'] * cand['length']

Gsimple = nx.Graph()
for row in cand[['u','v','length','weight']].itertuples(index=False):
    u, v, length, weight = row
    if Gsimple.has_edge(u, v):
        # 如果已有边，保留更大的 weight
        if weight > Gsimple[u][v]['weight']:
            Gsimple[u][v]['weight'] = weight
            Gsimple[u][v]['length'] = length
    else:
        Gsimple.add_edge(u, v, weight=weight, length=length)

# 候选为空直接退出
if Gsimple.number_of_edges() == 0:
    print("候选边为空（阈值过高）——请调低 Q")
else:
    # -------- 3) 最大生成树（每个连通分量各取一棵）--------
    # 对每个连通分量单独取 maximum spanning tree，再并起来
    T = nx.Graph()
    for comp_nodes in nx.connected_components(Gsimple):
        sub = Gsimple.subgraph(comp_nodes)
        T_comp = nx.maximum_spanning_tree(sub, weight='weight')  # 最大生成树
        T = nx.compose(T, T_comp)

    # -------- 4) 叶子修剪：直到达到“极简”目标 --------
    # 预计算一个权重阈值（分位）
    all_w = np.array([d['weight'] for _,_,d in T.edges(data=True)])
    leaf_weight_thr = np.quantile(all_w, LEAF_WEIGHT_Q) if len(all_w) else 0.0

    def total_km(G_):
        return sum(d.get('length', 0.0) for _,_,d in G_.edges(data=True)) / 1000.0  # 如果 length 单位为米

    iters = 0
    while total_km(T) > TARGET_TOTAL_KM and iters < MAX_ITERS:
        iters += 1
        # 找所有叶子节点（度=1）
        leaves = [n for n,deg in T.degree() if deg == 1]
        if not leaves:
            break

        # 候选叶子边（u--v，其中 u 是叶子）
        candidates = []
        for u in leaves:
            v = next(T.neighbors(u))
            data = T[u][v]
            candidates.append((u, v, data.get('weight', 0.0), data.get('length', 0.0)))

        if not candidates:
            break

        # 按“先剪短且低权重”的排序：权重升序、长度升序
        candidates.sort(key=lambda x: (x[2], x[3]))

        # 选择一个要剪的叶子边；尽量先剪低于分位阈值的
        cut_idx = 0
        for i, (u, v, w, L) in enumerate(candidates):
            if w <= leaf_weight_thr:
                cut_idx = i
                break

        u, v, _, _ = candidates[cut_idx]
        T.remove_node(u)   # 删掉叶子节点（等价于删这条叶子边）
        # 循环继续，直到总长度 <= 目标

    print(f"主干修剪完成：边数={T.number_of_edges()}, 总长度≈{total_km(T):.1f} km, 迭代={iters}")

    # -------- 5) 准备绘制主走廊 --------
    # 从 T 提取边列表并回到 GeoDataFrame
    corridors = []
    T_edges = list(T.edges())
    T_set = set(tuple(sorted(e)) for e in T_edges)  # 无向边的 (min(u),max(v)) 形式

    # 在 cand 里找与 T 对应的几何（cand 可能有多条同一对端点的边，选 weight 最大的）
    # 先做一个 (u,v) -> row 的最优索引
    best_edge = {}
    for row in cand[['u','v','geometry','weight','length']].itertuples():
        uv = tuple(sorted((row.u, row.v)))
        if (uv not in T_set):
            continue
        if (uv not in best_edge) or (row.weight > best_edge[uv]['weight']):
            best_edge[uv] = {'u': row.u, 'v': row.v, 'geometry': row.geometry,
                             'weight': row.weight, 'length': row.length}

    # 从 best_edge 构回 GeoDataFrame
    import geopandas as gpd

    if best_edge:
        corridors_gdf = pd.DataFrame(best_edge.values())
        corridors_gdf = gpd.GeoDataFrame(corridors_gdf, geometry='geometry', crs=edges_all.crs)

        # 给整列设置同一个颜色/线宽 —— 要按行数重复
        corridors_gdf['color'] = [(0, 0, 0, 1.0)] * len(corridors_gdf)
        corridors_gdf['linewidth'] = [3.0] * len(corridors_gdf)
    else:
        corridors_gdf = gpd.GeoDataFrame(columns=['u', 'v', 'geometry', 'weight', 'length', 'color', 'linewidth'],
                                         geometry='geometry', crs=edges_all.crs)

    # -------- 6) 画图 --------
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    # 背景淡化
    edges_gdf.plot(color=(0,0,0,0.08), linewidth=0.3, ax=ax3)
    # 主走廊
    if len(corridors_gdf):
        corridors_gdf = ox.utils_geo.graph_to_gdfs(T, nodes=False, edges=True) \
            if isinstance(T, (nx.Graph, nx.DiGraph)) and False else corridors_gdf  # 留作将来扩展
        # 直接用我们整理好的 corridors_gdf
        ax3.set_title(f"Ultra-sparse Bike Corridors (≤ {TARGET_TOTAL_KM} km)", fontsize=16)
        for _, r in corridors_gdf.iterrows():
            if isinstance(r['geometry'], LineString):
                xs, ys = r['geometry'].xy
                ax3.plot(xs, ys, linewidth=3.0, alpha=0.95)  # 简洁输出
            else:
                # 如果有 MultiLineString，逐段画
                try:
                    for geom in r['geometry'].geoms:
                        xs, ys = geom.xy
                        ax3.plot(xs, ys, linewidth=3.0, alpha=0.95)
                except Exception:
                    pass
    else:
        ax3.set_title("Ultra-sparse Bike Corridors (empty; try lower Q or higher TARGET_TOTAL_KM)", fontsize=16)

    fig3, ax3 = plt.subplots(figsize=(10, 10))
    edges_gdf.plot(color=(0, 0, 0, 0.08), linewidth=0.3, ax=ax3)

    if len(corridors_gdf):
        corridors_gdf.plot(color=corridors_gdf['color'],
                           linewidth=corridors_gdf['linewidth'], ax=ax3)

    ax3.set_axis_off()
    ax3.set_title(f"Ultra-sparse Bike Corridors (≤ {TARGET_TOTAL_KM} km)", fontsize=16)
    plt.tight_layout()
    plt.savefig("bike_corridors_ultra_sparse.png", dpi=300)
    plt.show()

# ========= 8. 把主干分量连起来（Steiner-like 连接） =========
import networkx as nx
import numpy as np

# ------ 参数 ------
LAMBDA = 2.0   # 连接时对低频边的惩罚强度（1~4 常用）
# 说明：cost = length * (1 + LAMBDA*(1 - f_norm))，f_norm∈[0,1]；频率高→成本低

# ------ A) 在全图上准备“cost”权重 ------
# 需要把 edges_gdf 的 freq/length 映射回 (u,v,key)
edges_all = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
edges_all = edges_all.merge(edges_gdf[['u','v','key','freq','length']], on=['u','v','key'], how='left')
edges_all['freq'] = edges_all['freq'].fillna(0)
edges_all['length'] = edges_all['length'].fillna(edges_all.geometry.length)

f_max = max(1e-9, edges_all['freq'].max())
edges_all['f_norm'] = edges_all['freq'] / f_max
edges_all['cost'] = edges_all['length'] * (1.0 + LAMBDA*(1.0 - edges_all['f_norm']))

# 建一个无向的“成本图”（多重边取最小 cost）
G_cost = nx.Graph()
for r in edges_all[['u','v','cost','length','freq']].itertuples(index=False):
    u, v, cost, length, freq = r
    if G_cost.has_edge(u, v):
        if cost < G_cost[u][v]['cost']:
            G_cost[u][v].update(dict(cost=cost, length=length, freq=freq))
    else:
        G_cost.add_edge(u, v, cost=cost, length=length, freq=freq)

# ------ B) 主干树 T 的连通分量（如果本来就连通就直接跳过） ------
comps = [set(c) for c in nx.connected_components(T)]
if len(comps) > 1:
    # 选每个分量的代表节点（简单取“度最大”的一个，也可换成介数中心等）
    reps = []
    for C in comps:
        sub = T.subgraph(C)
        rep = max(sub.degree, key=lambda x: x[1])[0]
        reps.append(rep)

    # ------ C) 组件层完全图：权重=组件代表之间在 G_cost 上的最短路成本 ------
    CG = nx.Graph()
    for i in range(len(reps)):
        CG.add_node(i, rep=reps[i])
    for i in range(len(reps)):
        for j in range(i+1, len(reps)):
            try:
                length_ij = nx.shortest_path_length(G_cost, reps[i], reps[j], weight='cost')
            except nx.NetworkXNoPath:
                length_ij = float('inf')
            CG.add_edge(i, j, cost=length_ij)

    # ------ D) 在组件层做 MST，得到需要连接的组件对 ------
    CG_mst = nx.minimum_spanning_tree(CG, weight='cost')

    # ------ E) 把这些组件对在 G_cost 上的最短路径加入 T（连接起来） ------
    T_connected = T.copy()
    paths_added = 0
    for u_c, v_c, d in CG_mst.edges(data=True):
        if not np.isfinite(d['cost']):
            continue
        u_rep, v_rep = CG.nodes[u_c]['rep'], CG.nodes[v_c]['rep']
        sp_nodes = nx.shortest_path(G_cost, u_rep, v_rep, weight='cost')
        # 把路径上的边都加进来
        for a, b in zip(sp_nodes[:-1], sp_nodes[1:]):
            if not T_connected.has_edge(a, b):
                T_connected.add_edge(a, b,
                                     weight=G_cost[a][b]['freq']*G_cost[a][b]['length'],
                                     length=G_cost[a][b]['length'])
                paths_added += 1
    print(f"连接完成：新增边 {paths_added} 条，分量 {len(comps)} -> {nx.number_connected_components(T_connected)}")

else:
    T_connected = T
    print("主干已连通，无需补全连接。")

# ------ F) 绘图：连通主干网 ------
# 从连通主干提取几何（用 edges_all 里 (u,v) 的最佳几何）
best_geom = {}
for r in edges_all[['u','v','geometry','freq','length']].itertuples(index=False):
    key_uv = tuple(sorted((r.u, r.v)))
    # 针对同一对端点，保留“频率*长度”最大的几何，便于连续外观
    val = r.freq * r.length
    if (key_uv not in best_geom) or (val > best_geom[key_uv]['score']):
        best_geom[key_uv] = dict(geometry=r.geometry, score=val)

lines = []
for a, b in T_connected.edges():
    uv = tuple(sorted((a, b)))
    g = best_geom.get(uv, None)
    if g is not None:
        lines.append(g['geometry'])

from geopandas import GeoDataFrame
corridors_connected = GeoDataFrame(geometry=lines, crs=edges_all.crs)

fig4, ax4 = plt.subplots(figsize=(10,10))
edges_gdf.plot(color=(0,0,0,0.07), linewidth=0.25, ax=ax4)
if len(corridors_connected):
    corridors_connected.plot(color='black', linewidth=3.2, ax=ax4)
ax4.set_axis_off()
ax4.set_title("Connected Ultra-sparse Bike Backbone", fontsize=16)
plt.tight_layout()
plt.savefig("bike_corridors_connected.png", dpi=300)
plt.show()

