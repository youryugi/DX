import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta
# 加载建筑物数据
building_gml_file =  r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357451_bldg_6697_op.gml"
building_gdf = gpd.read_file(building_gml_file)
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


import osmnx as ox
import networkx as nx

# 确保建筑物数据的坐标系为 WGS84（EPSG:4326）
if building_gdf.crs.to_epsg() != 4326:
    building_gdf = building_gdf.to_crs(epsg=4326)

# 获取建筑物边界范围
building_bounds = building_gdf.total_bounds  # (minx, miny, maxx, maxy)

# 使用 bbox 参数下载自行车道路数据
print('kaishi xiazai')
bbox = (building_bounds[3], building_bounds[1], building_bounds[2], building_bounds[0])  # (north, south, east, west)
print('xiazaiwancheng')
G = ox.graph_from_bbox(bbox=bbox, network_type="bike")
print(f"节点数量: {len(G.nodes)}")
print(f"边数量: {len(G.edges)}")
# 转换图为 GeoDataFrame
gdf_edges = ox.graph_to_gdfs(G, nodes=False)

# 转换自行车道路到 EPSG:6669（建筑物的坐标系）
if gdf_edges.crs.to_epsg() != 6669:
    gdf_edges = gdf_edges.to_crs(epsg=6669)



print('road')
# 动态路径规划功能
start_point = None
end_point = None

def on_click(event):
    global start_point, end_point
    if start_point is None:
        start_point = (event.xdata, event.ydata)
        print(f"起点已选择: {start_point}")
    elif end_point is None:
        end_point = (event.xdata, event.ydata)
        print(f"终点已选择: {end_point}")
        plot_path()


def plot_path():
    global start_point, end_point
    try:
        # 坐标转换逻辑
        start_point_transformed = ax.transData.inverted().transform(start_point)
        start_point_lonlat = (start_point_transformed[0], start_point_transformed[1])
        end_point_transformed = ax.transData.inverted().transform(end_point)
        end_point_lonlat = (end_point_transformed[0], end_point_transformed[1])

        # 最近节点匹配逻辑
        start_node = ox.nearest_nodes(G, start_point_lonlat[0], start_point_lonlat[1])
        end_node = ox.nearest_nodes(G, end_point_lonlat[0], end_point_lonlat[1])

        print(f"点击的起点 (EPSG:4326): {start_point_lonlat}")
        print(f"点击的终点 (EPSG:4326): {end_point_lonlat}")
        print(f"最近的起点节点: {G.nodes[start_node]}")
        print(f"最近的终点节点: {G.nodes[end_node]}")
        # 计算路径
        route = nx.shortest_path(G, source=start_node, target=end_node, weight="length")
        print(f"路径节点: {route}")

        # 提取路径边的几何
        route_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        lines = []
        for u, v in route_edges:
            edge_data = G[u][v][0]  # 获取边的几何
            geom = edge_data.get("geometry", None)
            if geom:
                lines.append(geom)
            else:
                # 如果没有几何信息，使用节点坐标构建线
                lines.append(LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                                         (G.nodes[v]["x"], G.nodes[v]["y"])]))

        # 创建 GeoDataFrame 并绘制到 ax
        route_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
        route_gdf = route_gdf.to_crs(epsg=6669)  # 转换到与 ax 一致的坐标系
        route_gdf.plot(ax=ax, color="red", linewidth=2, label="路径")

        plt.legend()
        plt.draw()  # 刷新现有窗口
        print("路径绘制完成")
    except nx.NetworkXNoPath:
        print("无法找到起点和终点之间的路径。")


# 动态交互，添加鼠标点击事件
fig.canvas.mpl_connect("button_press_event", on_click)

# 绘制阴影（灰色半透明）
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='阴影')

# 绘制建筑物（蓝色实线）
building_gdf.plot(ax=ax, color='lightblue', edgecolor='black', label='建筑物')

# 添加图例
legend_handles = [
    Patch(facecolor='lightblue', edgecolor='black', label='建筑物'),
    Patch(facecolor='gray', edgecolor='black', label='阴影'),
]

plt.title("建筑物及其地面阴影", fontsize=16)
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel("X 坐标 (米)")
plt.ylabel("Y 坐标 (米)")
plt.grid(True)

# 绘制自行车道路
gdf_edges.plot(ax=ax, color="blue", linewidth=1, label="自行车道路")
plt.show(block=True)  # 保证图形窗口保持打开
