import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
from matplotlib.patches import Patch

print("1. 正在读取建筑数据...")
bldg_gml_files = [
    r"bldg/51357451_bldg_6697_op.gml",
    # r"bldg\51357462_bldg_6697_op.gml",
    # r"bldg\51357463_bldg_6697_op.gml",
]
building_gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in bldg_gml_files], ignore_index=True))

# 只保留 usage 为 411、412、422
building_gdf['usage'] = building_gdf['usage'].astype(str).str.strip()
building_gdf = building_gdf[building_gdf['usage'].str.startswith(('411', '412', '422'))]

# 颜色映射
usage_color_map = {
    '411': '#32cbbf',   # Single-family house
    '412': '#984ea3',   # Apartment / Condominium
    '422': '#4daf4a',   # School (我们会筛学校)
}
default_color = '#cccccc'
building_gdf['color'] = building_gdf['usage'].apply(lambda u: usage_color_map.get(u[:3], default_color))

print("   建筑物总数：", len(building_gdf))
print("usage 字段唯一值（处理后）：", building_gdf['usage'].unique())

print("所有建筑物名字：")
possible_name_cols = ['name', 'bldNm', 'bldgName', '建物名称', '名称']
name_col = next((c for c in possible_name_cols if c in building_gdf.columns), None)
if name_col:
    print(building_gdf[name_col].tolist())
else:
    print("未找到'name'字段，实际字段有：", building_gdf.columns.tolist())

print("2. 计算建筑物范围并下载对应路网...")
# 转到 WGS84 以便从 bbox 下载路网
if building_gdf.crs is not None and building_gdf.crs.to_epsg() != 4326:
    building_gdf = building_gdf.to_crs(epsg=4326)
bounds = building_gdf.total_bounds  # [minx, miny, maxx, maxy]
minx, miny, maxx, maxy = bounds
padding = 0.001
minx -= padding; miny -= padding; maxx += padding; maxy += padding
print(f"   路网下载范围: ({miny:.5f}, {maxy:.5f}, {minx:.5f}, {maxx:.5f})")

bbox_str = f"{miny:.5f}_{maxy:.5f}_{minx:.5f}_{maxx:.5f}"
graphml_path = f"cached_network_{bbox_str}.graphml"

if os.path.exists(graphml_path):
    print(f"   检测到本地路网缓存文件 {graphml_path}，直接读取...")
    G = ox.load_graphml(graphml_path)
else:
    print("   未检测到缓存，开始下载路网...")
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='drive')
    ox.save_graphml(G, graphml_path)
    print(f"   路网已保存到 {graphml_path}")

print("   路网加载完成，节点数：", len(G.nodes), "，边数：", len(G.edges))

print("3. 正在查找 usage=411 / 412 / 422(学校) ...")
# 先筛 411/412
bldg_411_wgs = building_gdf[building_gdf['usage'].str.startswith('411')].copy()
bldg_412_wgs = building_gdf[building_gdf['usage'].str.startswith('412')].copy()

# 422 中只保留学校（名称关键字）
if name_col is None:
    raise ValueError("未找到可用的建筑名称字段，请在GML里确认字段名。")
school_keywords = [
    '学校','小学校','中学校','高校','大学','幼稚園','保育園',
    'School','College','University','Kindergarten','High School','Elementary School'
]
pattern = '|'.join(school_keywords)
bldg_422_wgs_all = building_gdf[building_gdf['usage'].str.startswith('422')].copy()
bldg_422_wgs = bldg_422_wgs_all[bldg_422_wgs_all[name_col].astype(str).str.contains(pattern, case=False, na=False)].copy()

print("   411 数量：", len(bldg_411_wgs))
print("   412 数量：", len(bldg_412_wgs))
print("   422(全部) 数量：", len(bldg_422_wgs_all))
print("   422(学校) 数量：", len(bldg_422_wgs))
if bldg_411_wgs.empty or bldg_422_wgs.empty:
    print("没有找到 411 或 学校(422)，程序终止。")
    exit()

print("4. 计算最短路径 & 统计边频率 ...")
edge_counter = Counter()
all_paths = []

# 投影到平面坐标系 EPSG:6669 做距离运算
projected_crs = 6669
building_proj = building_gdf.to_crs(epsg=projected_crs)
bldg_411 = building_proj[building_proj['usage'].str.startswith('411')]
bldg_412 = building_proj[building_proj['usage'].str.startswith('412')]
bldg_422 = building_proj[building_proj['usage'].str.startswith('422')]
bldg_422 = bldg_422[bldg_422[name_col].astype(str).str.contains(pattern, case=False, na=False)]

# 中心点
bldg_411_centroids = bldg_411.geometry.centroid
bldg_412_centroids = bldg_412.geometry.centroid
bldg_422_centroids = bldg_422.geometry.centroid

# ===== 411 -> 最近学校422 =====
for idx, c411 in enumerate(bldg_411_centroids):
    dists = bldg_422_centroids.distance(c411)
    n422_idx = dists.idxmin()
    c422 = bldg_422_centroids.loc[n422_idx]

    # 转回WGS84找最近路网节点
    c411_wgs = gpd.GeoSeries([c411], crs=projected_crs).to_crs(epsg=4326).iloc[0]
    c422_wgs = gpd.GeoSeries([c422], crs=projected_crs).to_crs(epsg=4326).iloc[0]
    orig = ox.distance.nearest_nodes(G, X=c411_wgs.x, Y=c411_wgs.y)
    dest = ox.distance.nearest_nodes(G, X=c422_wgs.x, Y=c422_wgs.y)

    try:
        path = nx.shortest_path(G, source=orig, target=dest, weight='length')
        all_paths.append(path)
        for u, v in zip(path[:-1], path[1:]):
            edge_counter[(u, v) if G.has_edge(u, v) else (v, u)] += 1
        if (idx+1) % 100 == 0 or (idx+1) == len(bldg_411_centroids):
            print(f"   已处理 {idx+1}/{len(bldg_411_centroids)} 个 411")
    except nx.NetworkXNoPath:
        print(f"   第{idx+1}个 411 与最近学校 422 间无路径")
        continue

# ===== 412 -> 最近学校422（频率 * 22.5）=====
for idx, c412 in enumerate(bldg_412_centroids):
    dists = bldg_422_centroids.distance(c412)
    n422_idx = dists.idxmin()
    c422 = bldg_422_centroids.loc[n422_idx]

    c412_wgs = gpd.GeoSeries([c412], crs=projected_crs).to_crs(epsg=4326).iloc[0]
    c422_wgs = gpd.GeoSeries([c422], crs=projected_crs).to_crs(epsg=4326).iloc[0]
    orig = ox.distance.nearest_nodes(G, X=c412_wgs.x, Y=c412_wgs.y)
    dest = ox.distance.nearest_nodes(G, X=c422_wgs.x, Y=c422_wgs.y)

    try:
        path = nx.shortest_path(G, source=orig, target=dest, weight='length')
        for u, v in zip(path[:-1], path[1:]):
            edge_counter[(u, v) if G.has_edge(u, v) else (v, u)] += 22.5
        if (idx+1) % 100 == 0 or (idx+1) == len(bldg_412_centroids):
            print(f"   已处理 {idx+1}/{len(bldg_412_centroids)} 个 412")
    except nx.NetworkXNoPath:
        print(f"   第{idx+1}个 412 与最近学校 422 间无路径")
        continue

# 路网投影
if G.graph.get('crs') != 'epsg:6669':
    G_proj = ox.project_graph(G, to_crs='epsg:6669')
else:
    G_proj = G

# ===== 5. 可视化（411、412 都可视化；422 只画学校）=====
fig, ax = plt.subplots(figsize=(12, 8))
ox.plot_graph(G_proj, ax=ax, show=False, close=False,
              edge_color='lightgray', node_size=0, edge_linewidth=0.5)

#（可选）道路
if 'road_gdf' in locals():
    road_gdf = road_gdf.to_crs(epsg=6669)
    road_gdf.plot(ax=ax, color='black', linewidth=1, alpha=0.5, label='Road')

# 把三个类别分别 plot
def ensure_color(df):
    if 'color' not in df.columns:
        df['color'] = df['usage'].str[:3].map(usage_color_map).fillna(default_color)
    return df

house_gdf  = ensure_color(bldg_411.copy())
apt_gdf    = ensure_color(bldg_412.copy())
school_gdf = ensure_color(bldg_422.copy())  # 已是学校

house_gdf.plot(ax=ax, color=house_gdf['color'], edgecolor=None, linewidth=0, alpha=0.9)
apt_gdf.plot(ax=ax,   color=apt_gdf['color'],   edgecolor=None, linewidth=0, alpha=0.9)
school_gdf.plot(ax=ax,color=school_gdf['color'],edgecolor=None, linewidth=0, alpha=0.9)

# 画路径频率
max_freq = max(edge_counter.values()) if edge_counter else 1
for (u, v), freq in edge_counter.items():
    data = G_proj.get_edge_data(u, v)
    if data:
        geom = data[0].get('geometry')
    else:
        geom = None
    if geom is not None:
        xs, ys = geom.xy
    else:
        xs = [G_proj.nodes[u]['x'], G_proj.nodes[v]['x']]
        ys = [G_proj.nodes[u]['y'], G_proj.nodes[v]['y']]
    ax.plot(xs, ys,
            color='red',
            linewidth=1 + 4 * freq / max_freq,
            alpha=0.8)

# 图例（只列出 411/412/422(学校)）
legend_handles = [
    Patch(facecolor=usage_color_map['411'], label='Single-family house'),
    Patch(facecolor=usage_color_map['412'], label='Apartment / Condominium'),
    Patch(facecolor=usage_color_map['422'], label='School'),
]
plt.legend(handles=legend_handles, loc='upper right', fontsize=12)

plt.title("Shortest Path Edge Frequency (Schools for 422 only)", fontsize=16)
plt.tight_layout()
plt.show()
print("全部完成！")

# ---------- 保存边频率 ----------
print("5. 保存边频率……")
freq_records = []
for (u, v), freq in edge_counter.items():
    data = G_proj.get_edge_data(u, v)
    geom = data[0].get('geometry') if data else None
    freq_records.append({'u': u, 'v': v, 'freq': freq, 'geometry': geom})

freq_gdf = gpd.GeoDataFrame(freq_records, geometry='geometry', crs='EPSG:6669')
freq_tag = os.path.splitext(graphml_path)[0].replace('cached_network_', '')
gpkg_path = f"edge_freq_home-school{freq_tag}.gpkg"
csv_path  = f"edge_freq_home-school{freq_tag}.csv"

freq_gdf.to_file(gpkg_path, layer='edge_freq', driver='GPKG')
freq_gdf.drop(columns='geometry').to_csv(csv_path, index=False)
print(f"   已保存到 {gpkg_path} 与 {csv_path}")
