import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta
import osmnx as ox
import networkx as nx
from shapely.ops import unary_union

# -------------------------
# 数据加载与坐标处理
# -------------------------
building_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357451_bldg_6697_op.gml"
road_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\tran\51357451_tran_6697_op.gml"

building_gdf = gpd.read_file(building_gml_file)
road_gdf = gpd.read_file(road_gml_file)

# 确保建筑物和道路使用 EPSG:6669 (假设为当地投影坐标系) 以计算阴影
if building_gdf.crs.to_epsg() != 6669:
    building_gdf = building_gdf.to_crs(epsg=6669)
if road_gdf.crs.to_epsg() != 6669:
    road_gdf = road_gdf.to_crs(epsg=6669)

# -------------------------
# 计算太阳高度角和方位角
# -------------------------
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)
date_time = datetime(2024, 12, 5, 13, 10, tzinfo=timezone(timedelta(hours=9)))

solar_elevation = elevation(city.observer, date_time)
solar_azimuth = azimuth(city.observer, date_time)

print(f"太阳高度角: {solar_elevation:.2f}°")
print(f"太阳方位角: {solar_azimuth:.2f}°")

if solar_elevation <= 0:
    print("太阳位于地平线以下，无法生成阴影。")
    exit()

sun_vector = np.array([
    np.cos(np.radians(solar_elevation)) * np.sin(np.radians(solar_azimuth)),
    np.cos(np.radians(solar_elevation)) * np.cos(np.radians(solar_azimuth)),
    np.sin(np.radians(solar_elevation))
])

# 查找高度列
height_column = None
for col in building_gdf.columns:
    if 'height' in col.lower():
        height_column = col
        break

if height_column is not None:
    building_gdf[height_column] = building_gdf[height_column].fillna(3)
else:
    print("没有找到高度列，将默认高度设为3米用于计算阴影。")
    height_column = 'default_height'
    building_gdf[height_column] = 3.0

# -------------------------
# 阴影计算函数
# -------------------------
def shadow_using_lines(geometry, height):
    if sun_vector[2] <= 0:
        return None
    polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
    shadow_lines = []

    for poly in polygons:
        base_coords = [(x, y) for x, y, *rest in poly.exterior.coords]
        shadow_coords = [
            (
                x - height / np.tan(np.radians(solar_elevation)) * sun_vector[0],
                y - height / np.tan(np.radians(solar_elevation)) * sun_vector[1]
            )
            for x, y in base_coords
        ]

        for base, shadow in zip(base_coords, shadow_coords):
            shadow_lines.append(LineString([base, shadow]))

    union_lines = unary_union(shadow_lines)
    shadow_polygon = union_lines.convex_hull
    return shadow_polygon

# -------------------------
# 为每个建筑物生成阴影
# -------------------------
building_gdf['shadow'] = building_gdf.apply(
    lambda row: shadow_using_lines(row.geometry, row[height_column]), axis=1
)

if 'shadow' not in building_gdf.columns or building_gdf['shadow'].isnull().all():
    print("未生成任何有效阴影。请检查数据或逻辑。")
    exit()

shadow_gdf = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')

# -------------------------
# 准备绘图
# -------------------------
plt.rcParams['font.family'] = 'SimHei'
fig, ax = plt.subplots(figsize=(12, 8))

bounds = building_gdf.total_bounds
buffer = 10
x_min, y_min, x_max, y_max = bounds
x_min -= buffer
y_min -= buffer
x_max += buffer
y_max += buffer

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
road_gdf.plot(ax=ax, color='yellow', alpha=0.5,label='road')
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='shadow')
building_gdf.plot(ax=ax, color='lightblue',  label='building')

legend_handles = [
    Patch(facecolor='yellow', label='Road'),
    Patch(facecolor='lightblue', label='Building'),
    Patch(facecolor='gray', edgecolor='black', label='Shadow'),
]
plt.title("Roads, buildings, and shadows     ↑North", fontsize=16)
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)

# -------------------------
# 获取路网和路线规划
# -------------------------
# 在下载OSM数据时需要WGS84
# 因为building_gdf此时是EPSG:6669，我们先转换为WGS84获取bbox
building_gdf_wgs84 = building_gdf.to_crs(epsg=4326)
building_bounds_wgs84 = building_gdf_wgs84.total_bounds  # (minx, miny, maxx, maxy)

# bbox顺序：(北, 南, 东, 西)
bbox = (building_bounds_wgs84[3], building_bounds_wgs84[1],
        building_bounds_wgs84[2], building_bounds_wgs84[0])
print('kaishi xiazai')
print(bbox)
G = ox.graph_from_bbox(north=bbox[0], south=bbox[1], east=bbox[2], west=bbox[3], network_type="bike")
print('xiazaiwancheng')
print(f"节点数量: {len(G.nodes)}")
print(f"边数量: {len(G.edges)}")

# 将图转换为GeoDataFrame，并投影为EPSG:6669以计算长度和阴影覆盖
gdf_edges = ox.graph_to_gdfs(G, nodes=False)
if gdf_edges.crs.to_epsg() != 4326:
    gdf_edges = gdf_edges.to_crs(epsg=4326)
# 先转为EPSG:6669以便长度测量和阴影计算
gdf_edges = gdf_edges.to_crs(epsg=6669)

# 原始路径规划（基于长度）
origin_point = (34.6266, 135.5133)  # (lat, lon)
destination_point = (34.6390, 135.5190)  # (lat, lon)

orig_node = ox.distance.nearest_nodes(G, X=origin_point[1], Y=origin_point[0])
dest_node = ox.distance.nearest_nodes(G, X=destination_point[1], Y=destination_point[0])

route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')

# route_edges需要(u,v,key)
# 默认key=0，若无平行边
route_edges = [(route[i], route[i+1], 0) for i in range(len(route)-1)]
route_gdf = gdf_edges.loc[route_edges]

# 将route_gdf在同一EPSG:6669下绘制
route_gdf.plot(ax=ax, color='red', linewidth=2, label='Shortest Bike Route')

# -------------------------
# 计算阴凉权重并规划最凉爽路线
# -------------------------
shadow_union = unary_union(shadow_gdf.geometry)

shadow_lengths = []
for idx, row in gdf_edges.iterrows():
    edge_geom = row.geometry
    edge_length = edge_geom.length
    intersection_geom = edge_geom.intersection(shadow_union)
    shadowed_length = intersection_geom.length if not intersection_geom.is_empty else 0
    shadow_ratio = shadowed_length / edge_length if edge_length > 0 else 0
    # 定义权重：cost = length * (1 - shadow_ratio)，阴影越多cost越低
    cost = edge_length * (1 - shadow_ratio)
    shadow_lengths.append(cost)

gdf_edges['cool_weight'] = shadow_lengths

# 将cool_weight写回G
for (u,v,k), val in zip(gdf_edges.index, gdf_edges['cool_weight']):
    if (u,v,k) in G.edges:
        G[u][v][k]['cool_weight'] = val

cool_route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='cool_weight')
cool_route_edges = [(cool_route[i], cool_route[i+1], 0) for i in range(len(cool_route)-1)]
cool_route_gdf = gdf_edges.loc[cool_route_edges]
# 为每个几何体应用一个小的平移，比如向x方向平移0.5米
from shapely.affinity import translate
cool_route_gdf['geometry'] = cool_route_gdf.geometry.apply(lambda g: translate(g, xoff=1.5, yoff=1.5))

cool_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Coolest Bike Route')
plt.legend()
plt.show()
