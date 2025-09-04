import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from shapely.ops import nearest_points
import os
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

print("1. 正在读取建筑数据...")
bldg_gml_files = [
        r"bldg/51357451_bldg_6697_op.gml",
        #r"bldg\51357462_bldg_6697_op.gml",
        #r"bldg\51357463_bldg_6697_op.gml",
    # r"bldg\51357451_bldg_6697_op.gml",
    # r"bldg\51357452_bldg_6697_op.gml",
    # r"bldg\51357453_bldg_6697_op.gml",
    # r"bldg\51357461_bldg_6697_op.gml",
    # r"bldg\51357462_bldg_6697_op.gml",
    # r"bldg\51357463_bldg_6697_op.gml",
    # r"bldg\51357471_bldg_6697_op.gml",
    # r"bldg\51357472_bldg_6697_op.gml",
    # r"bldg\51357473_bldg_6697_op.gml"
]
building_gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in bldg_gml_files], ignore_index=True))
# 只保留 usage 为 411、412、422 的建筑
building_gdf['usage'] = building_gdf['usage'].astype(str).str.strip()
building_gdf = building_gdf[building_gdf['usage'].str.startswith(('411', '412', '422'))]

# 颜色映射
usage_color_map = {
    '411': '#32cbbf',   # 绿色#ff7f00
    '412': "#984ea3",   # 紫色
    '422': '#4daf4a',   # 橙色
}
default_color = '#cccccc'
building_gdf['color'] = building_gdf['usage'].apply(lambda u: usage_color_map.get(u[:3], default_color))

print("   建筑物总数：", len(building_gdf))
print("usage 字段唯一值（处理后）：", building_gdf['usage'].unique())

print("所有建筑物名字：")
if 'name' in building_gdf.columns:
    print(building_gdf['name'].tolist())
else:
    print("未找到'name'字段，实际字段有：", building_gdf.columns.tolist())

print("2. 计算建筑物范围并下载对应路网...")
# 统一投影为WGS84
if building_gdf.crs is not None and building_gdf.crs.to_epsg() != 4326:
    building_gdf = building_gdf.to_crs(epsg=4326)
bounds = building_gdf.total_bounds  # [minx, miny, maxx, maxy]
minx, miny, maxx, maxy = bounds
padding = 0.001  # 可适当加大范围，单位为经纬度
minx -= padding
miny -= padding
maxx += padding
maxy += padding
print(f"   路网下载范围: ({miny:.5f}, {maxy:.5f}, {minx:.5f}, {maxx:.5f})")

# 生成唯一的缓存文件名（保留5位小数，防止文件名过长）
bbox_str = f"{miny:.5f}_{maxy:.5f}_{minx:.5f}_{maxx:.5f}"
graphml_path = f"cached_network_{bbox_str}.graphml"

if os.path.exists(graphml_path):
    print(f"   检测到本地路网缓存文件 {graphml_path}，直接读取...")
    G = ox.load_graphml(graphml_path)
else:
    print(f"   未检测到缓存，开始下载路网...")
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='drive')
    ox.save_graphml(G, graphml_path)
    print(f"   路网已保存到 {graphml_path}")

print("   路网加载完成，节点数：", len(G.nodes), "，边数：", len(G.edges))

print("3. 正在查找 usage=411 和 usage=422 的建筑...")
bldg_411 = building_gdf[building_gdf['usage'].str.startswith('411')]
bldg_422 = building_gdf[building_gdf['usage'].str.startswith('422')]
print("   usage=411 的建筑数：", len(bldg_411))
print("   usage=422 的建筑数：", len(bldg_422))
if bldg_411.empty or bldg_422.empty:
    print("没有找到 usage=411 或 usage=422 的建筑，程序终止。")
    exit()

b422 = bldg_422.iloc[0]

print("4. 正在计算所有最短路径并统计边频率（每个411只连最近的422）...")

edge_counter = Counter()
all_paths = []

# 先获取所有422建筑的中心点
bldg_422_centroids = bldg_422.geometry.centroid

# 投影到平面坐标系用于空间分析（如 EPSG:6669 或 UTM zone ）
projected_crs = 6669
if building_gdf.crs is None or building_gdf.crs.to_epsg() != projected_crs:
    building_gdf = building_gdf.to_crs(epsg=projected_crs)

bldg_411 = building_gdf[building_gdf['usage'].str.startswith('411')]
bldg_422 = building_gdf[building_gdf['usage'].str.startswith('422')]

# 计算中心点
bldg_411_centroids = bldg_411.geometry.centroid
bldg_422_centroids = bldg_422.geometry.centroid

# 记录所有配对的422中心点
paired_422_x = []
paired_422_y = []

# ===== 411→最近422 =====
for idx, b411_centroid in enumerate(bldg_411_centroids):
    # 找最近的422建筑
    distances = bldg_422_centroids.distance(b411_centroid)
    nearest_422_idx = distances.idxmin()
    nearest_422_centroid = bldg_422_centroids.loc[nearest_422_idx]
    paired_422_x.append(nearest_422_centroid.x)
    paired_422_y.append(nearest_422_centroid.y)

    # 匹配到路网节点（注意：ox.nearest_nodes 需要WGS84坐标）
    b411_centroid_wgs = gpd.GeoSeries([b411_centroid], crs=projected_crs).to_crs(epsg=4326).iloc[0]
    nearest_422_centroid_wgs = gpd.GeoSeries([nearest_422_centroid], crs=projected_crs).to_crs(epsg=4326).iloc[0]
    orig_node = ox.distance.nearest_nodes(G, X=b411_centroid_wgs.x, Y=b411_centroid_wgs.y)
    dest_node = ox.distance.nearest_nodes(G, X=nearest_422_centroid_wgs.x, Y=nearest_422_centroid_wgs.y)

    try:
        path = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
        all_paths.append(path)
        for u, v in zip(path[:-1], path[1:]):
            edge = (u, v) if G.has_edge(u, v) else (v, u)
            edge_counter[edge] += 1
        if (idx + 1) % 100 == 0 or (idx + 1) == len(bldg_411):
            print(f"   已处理 {idx+1}/{len(bldg_411)} 个411建筑")
    except nx.NetworkXNoPath:
        print(f"   第{idx+1}个411建筑与最近的422建筑之间无路径")
        continue

# ===== 412→最近422，频率乘以22.5 =====
bldg_412 = building_gdf[building_gdf['usage'].str.startswith('412')]
bldg_412_centroids = bldg_412.geometry.centroid
for idx, b412_centroid in enumerate(bldg_412_centroids):
    distances = bldg_422_centroids.distance(b412_centroid)
    nearest_422_idx = distances.idxmin()
    nearest_422_centroid = bldg_422_centroids.loc[nearest_422_idx]

    b412_centroid_wgs = gpd.GeoSeries([b412_centroid], crs=projected_crs).to_crs(epsg=4326).iloc[0]
    nearest_422_centroid_wgs = gpd.GeoSeries([nearest_422_centroid], crs=projected_crs).to_crs(epsg=4326).iloc[0]
    orig_node = ox.distance.nearest_nodes(G, X=b412_centroid_wgs.x, Y=b412_centroid_wgs.y)
    dest_node = ox.distance.nearest_nodes(G, X=nearest_422_centroid_wgs.x, Y=nearest_422_centroid_wgs.y)

    try:
        path = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
        for u, v in zip(path[:-1], path[1:]):
            edge = (u, v) if G.has_edge(u, v) else (v, u)
            edge_counter[edge] += 22.5  # 频率乘以22.5
        if (idx + 1) % 100 == 0 or (idx + 1) == len(bldg_412_centroids):
            print(f"   已处理 {idx+1}/{len(bldg_412_centroids)} 个412建筑")
    except nx.NetworkXNoPath:
        print(f"   第{idx+1}个412建筑与最近的422建筑之间无路径")
        continue

# 路网投影到 EPSG:6669
if G.graph.get('crs') != 'epsg:6669':
    G_proj = ox.project_graph(G, to_crs='epsg:6669')
else:
    G_proj = G

# 可视化
fig, ax = plt.subplots(figsize=(12, 8))
ox.plot_graph(G_proj, ax=ax, show=False, close=False, edge_color='lightgray', node_size=0, edge_linewidth=0.5)

# if 'road_gdf' in locals():
#     road_gdf = road_gdf.to_crs(epsg=6669)
#     road_gdf.plot(ax=ax, color='black', linewidth=1, alpha=0.5, label='Road')

building_gdf.plot(
    ax=ax,
    color=building_gdf['color'],
    edgecolor=None,      # 没有黑边
    linewidth=0,
    alpha=0.9
)

# 3. 画路径频率线
max_freq = max(edge_counter.values()) if edge_counter else 1

for (u, v), freq in edge_counter.items():
    data = G_proj.get_edge_data(u, v)  # <- 用投影后的图
    if data:
        geom = data[0].get('geometry')
        if geom is not None:
            xs, ys = geom.xy
        else:
            xs = [G_proj.nodes[u]['x'], G_proj.nodes[v]['x']]
            ys = [G_proj.nodes[u]['y'], G_proj.nodes[v]['y']]
        ax.plot(xs, ys,
                color='red',
                linewidth=1 + 4 * freq / max_freq,
                alpha=0.8)


# 4. 图例
legend_handles = [
    Patch(facecolor=usage_color_map['411'], label='Single-family house'),
    Patch(facecolor=usage_color_map['412'], label='Apartment / Condominium'),
    Patch(facecolor=usage_color_map['422'], label='School'),
    #Patch(facecolor='black', label='Road')
]
plt.legend(handles=legend_handles, loc='upper right', fontsize=12)

plt.title("Shortest Path Edge Frequency and Building Usage", fontsize=16)
plt.tight_layout()
plt.show()
print("全部完成！")

# ---------- 新增：保存 edge_counter 为文件 ----------
print("5. 保存边频率……")
freq_records = []
for (u, v), freq in edge_counter.items():
    # 取投影后图中的几何；若无则用直线
    data = G_proj.get_edge_data(u, v)
    if data:
        geom = data[0].get('geometry')
    else:
        geom = None
    freq_records.append({
        'u': u,
        'v': v,
        'freq': freq,
        'geometry': geom
    })

# 转为 GeoDataFrame（若缺少几何也没问题）
freq_gdf = gpd.GeoDataFrame(freq_records, geometry='geometry', crs='EPSG:6669')

# 生成简短文件名
freq_tag = os.path.splitext(graphml_path)[0].replace('cached_network_', '')
gpkg_path = f"edge_freq_home-school{freq_tag}.gpkg"
csv_path  = f"edge_freq_home-school{freq_tag}.csv"

# 保存
freq_gdf.to_file(gpkg_path, layer='edge_freq', driver='GPKG')
freq_gdf.drop(columns='geometry').to_csv(csv_path, index=False)
print(f"   已保存到 {gpkg_path} 与 {csv_path}")
# ---------------------------------------------------
