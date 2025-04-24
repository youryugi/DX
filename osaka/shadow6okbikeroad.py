import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta
import osmnx as ox
import networkx as nx
# 加载建筑物数据
building_gml_file =  r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357451_bldg_6697_op.gml"
building_gdf = gpd.read_file(building_gml_file)
road_gml_file =  r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\tran\51357451_tran_6697_op.gml"
road_gdf = gpd.read_file(road_gml_file)
# 设置大阪的位置和时间
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)
date_time = datetime(2024, 12, 1, 13, 10, tzinfo=timezone(timedelta(hours=9)))

# 计算太阳高度角和方位角
solar_elevation = elevation(city.observer, date_time)
solar_azimuth = azimuth(city.observer, date_time)

print(f"太阳高度角: {solar_elevation:.2f}°")
print(f"太阳方位角: {solar_azimuth:.2f}°")

if solar_elevation <= 0:
    print("太阳位于地平线以下，无法生成阴影。")
    exit()

# 计算太阳向量
sun_vector = np.array([
    np.cos(np.radians(solar_elevation)) * np.sin(np.radians(solar_azimuth)),
    np.cos(np.radians(solar_elevation)) * np.cos(np.radians(solar_azimuth)),
    np.sin(np.radians(solar_elevation))
])

# 确保建筑物坐标系一致
if building_gdf.crs.to_epsg() != 6669:
    print("Reprojecting building data to EPSG:6669 for accurate shadow calculation...")
    building_gdf = building_gdf.to_crs(epsg=6669)
# 确保建筑物坐标系一致
if road_gdf.crs.to_epsg() != 6669:
    print("Reprojecting road data to EPSG:6669 for accurate shadow calculation...")
    road_gdf = road_gdf.to_crs(epsg=6669)

# 确定高度列
height_column = None
for col in building_gdf.columns:
    if 'height' in col.lower():
        height_column = col
        break

if height_column is not None:
    building_gdf[height_column] = building_gdf[height_column].fillna(3)  # 默认高度为 3 米
else:
    print("没有找到高度列，将在后续处理中使用默认高度 3 米。")

# 精确的阴影生成函数
from shapely.geometry import LineString
from shapely.ops import unary_union

def shadow_using_lines(geometry, height):
    if sun_vector[2] <= 0:
        print("太阳位于地平线以下，不生成阴影。")
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

        # 使用 LineString 连接底部和阴影顶点
        for base, shadow in zip(base_coords, shadow_coords):
            shadow_lines.append(LineString([base, shadow]))

    # 合并线并生成包络
    union_lines = unary_union(shadow_lines)
    shadow_polygon = union_lines.convex_hull  # 或者使用 buffer

    return shadow_polygon

# 为每个建筑物生成阴影
default_height = 3
building_gdf['shadow'] = building_gdf.apply(
    lambda row: shadow_using_lines(
        row.geometry,
        row[height_column] if height_column else default_height
    ), axis=1
)

# 检查是否生成了阴影
if 'shadow' not in building_gdf.columns or building_gdf['shadow'].isnull().all():
    print("未生成任何有效阴影。请检查数据或代码逻辑。")
    exit()

# 准备用于可视化的 GeoDataFrame
shadow_gdf = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')

# 可视化建筑物和阴影
plt.rcParams['font.family'] = 'SimHei'
fig, ax = plt.subplots(figsize=(12, 8))

# 设置绘图范围
bounds = building_gdf.total_bounds
buffer = 10
x_min, y_min, x_max, y_max = bounds
x_min -= buffer
y_min -= buffer
x_max += buffer
y_max += buffer

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
#
road_gdf.plot(ax=ax, color='yellow', edgecolor='black', label='road')
# 绘制阴影（灰色半透明）
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='shadow')

# 绘制建筑物（蓝色实线）
building_gdf.plot(ax=ax, color='lightblue', edgecolor='black', label='building')
# 添加图例
legend_handles = [
    Patch(facecolor='yellow', edgecolor='black', label='Road'),
    Patch(facecolor='lightblue', edgecolor='black', label='Building'),
    Patch(facecolor='gray', edgecolor='black', label='Shadow'),
]

plt.title("Roads, buildings, and shadows     ↑North", fontsize=16)
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)


# 确保建筑物数据的坐标系为 WGS84（EPSG:4326）
if building_gdf.crs.to_epsg() != 4326:
    building_gdf = building_gdf.to_crs(epsg=4326)

# 获取建筑物边界范围
building_bounds = building_gdf.total_bounds  # (minx, miny, maxx, maxy)

# 使用 bbox 参数下载自行车道路数据
print('kaishi xiazai')
bbox = (building_bounds[3], building_bounds[1], building_bounds[2], building_bounds[0])  # (north, south, east, west)
print(bbox)
print('xiazaiwancheng')
G = ox.graph_from_bbox(bbox=bbox, network_type="bike")
print(f"节点数量: {len(G.nodes)}")
print(f"边数量: {len(G.edges)}")
# 转换图为 GeoDataFrame
gdf_edges = ox.graph_to_gdfs(G, nodes=False)

# 转换自行车道路到 EPSG:6669（建筑物的坐标系）
if gdf_edges.crs.to_epsg() != 6669:
    gdf_edges = gdf_edges.to_crs(epsg=6669)

#gdf_edges.plot(ax=ax, color="blue", linewidth=0.5, label="自行车道路")


#(34.63341667893838, 34.62494602643529, 135.52505045269916, 135.5123808328594)路径规划范围
origin_point = (34.6266, 135.5133)  # 替换为您实际的起点坐标（纬度, 经度）
destination_point = (34.6390, 135.5190)  # 替换为您实际的终点坐标

orig_node = ox.distance.nearest_nodes(G, X=origin_point[1], Y=origin_point[0])  # 注意：函数参数顺序为 lon, lat
dest_node = ox.distance.nearest_nodes(G, X=destination_point[1], Y=destination_point[0])
print(orig_node)
print(dest_node)
route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
route_edges = list(zip(route[:-1], route[1:]))
# 利用graph_to_gdfs获取边GDF
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
route_gdf = gdf_edges.loc[gdf_edges.index.isin(route_edges)]
if route_gdf.crs.to_epsg() != 6669:
    route_gdf = route_gdf.to_crs(epsg=6669)
# 若要在您现有的 matplotlib 画布 ax 上绘制路径：
route_gdf.plot(ax=ax, color='red', linewidth=2, label='Shortest Bike Route')
plt.legend()

import geopandas as gpd
from shapely.ops import unary_union

# 假设gdf_edges是您的自行车道路数据，shadow_gdf是阴影多边形数据
# gdf_edges 和 shadow_gdf 都应在同一投影坐标系下（如EPSG:4326或EPSG:6669）
# 如果不一致，请进行to_crs转换

# 将所有阴影多边形合并成一个大的多边形集，减少重复计算
shadow_union = unary_union(shadow_gdf.geometry)

# 添加新列：阴影覆盖长度比和权重
shadow_lengths = []
for idx, row in gdf_edges.iterrows():
    edge_geom = row.geometry
    edge_length = edge_geom.length  # 根据投影坐标系这里是米，否则在EPSG:4326下为度数，需要转换长度单位
    intersection_geom = edge_geom.intersection(shadow_union)
    shadowed_length = intersection_geom.length if not intersection_geom.is_empty else 0

    shadow_ratio = shadowed_length / edge_length if edge_length > 0 else 0

    # 定义权重，例如：cost = edge_length * (1 - shadow_ratio)
    # 阴影越多，(1 - shadow_ratio)越小 -> cost更低
    cost = edge_length * (1 - shadow_ratio)

    shadow_lengths.append(cost)

gdf_edges['cool_weight'] = shadow_lengths

# 将这个属性回写到G中
# 我们需要使用与G中edge对应的keys进行匹配（如果是MultiDiGraph，需要edge keys）
# 通常情况下 gdf_edges.index 会是 (u,v,key) 的多级索引
for (u, v, k), cost_val in zip(gdf_edges.index, gdf_edges['cool_weight']):
    # 为每条边添加cool_weight属性
    G[u][v][k]['cool_weight'] = cost_val

# 现在G的每条边有一个cool_weight的属性，使用该属性计算最阴凉的路线
cool_route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='cool_weight')
cool_route_edges = list(zip(cool_route[:-1], cool_route[1:]))
cool_route_gdf = gdf_edges.loc[cool_route_edges]


# 绘制"最凉爽"路线
cool_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Coolest Bike Route')

plt.legend()
plt.show()

plt.show()