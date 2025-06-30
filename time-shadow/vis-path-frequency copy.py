import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from shapely.ops import nearest_points

print("1. 正在读取建筑数据...")
bldg_gml_files = [
        r"bldg\51357462_bldg_6697_op.gml",
        r"bldg\51357463_bldg_6697_op.gml",
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
building_gdf['usage'] = building_gdf['usage'].astype(str).str.strip()
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
G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='drive')
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
        if (idx + 1) % 10 == 0 or (idx + 1) == len(bldg_411):
            print(f"   已处理 {idx+1}/{len(bldg_411)} 个411建筑")
    except nx.NetworkXNoPath:
        print(f"   第{idx+1}个411建筑与最近的422建筑之间无路径")
        continue

print("5. 正在可视化结果...")
fig, ax = plt.subplots(figsize=(12, 8))
ox.plot_graph(G, ax=ax, show=False, close=False, edge_color='lightgray', node_size=0, edge_linewidth=0.5)

max_freq = max(edge_counter.values()) if edge_counter else 1
for (u, v), freq in edge_counter.items():
    data = G.get_edge_data(u, v)
    if data:
        geom = data[0].get('geometry', None)
        if geom is not None:
            xs, ys = geom.xy
        else:
            xs = [G.nodes[u]['x'], G.nodes[v]['x']]
            ys = [G.nodes[u]['y'], G.nodes[v]['y']]
        ax.plot(xs, ys, color='red', linewidth=1 + 4 * freq / max_freq, alpha=0.8)

# 画所有411建筑中心点
ax.scatter(bldg_411_centroids.x, bldg_411_centroids.y, color='green', s=20, label='411 buildings')
# 画所有配对过的422建筑中心点
ax.scatter(paired_422_x, paired_422_y, color='blue', s=20, label='Paired 422 buildings')
# 画所有422建筑中心点（橙色）
ax.scatter(
    bldg_422_centroids.x, bldg_422_centroids.y,
    color='orange', s=120, marker='*', label='All 422 buildings',
    edgecolors='black', linewidths=1, zorder=10
)

plt.title("Shortest Path Edge Frequency (411→422)")
plt.legend()
plt.show()
print("全部完成！")