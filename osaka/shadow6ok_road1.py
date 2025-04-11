import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.ops import unary_union
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta

# 加载建筑物数据
building_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357451_bldg_6697_op.gml"
building_gdf = gpd.read_file(building_gml_file)

# 设置大阪的位置和时间
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)
date_time = datetime(2024, 12, 5, 14, 10, tzinfo=timezone(timedelta(hours=9)))

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

# 确保建筑物使用 EPSG:6669
if building_gdf.crs.to_epsg() != 6669:
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
def shadow_using_lines(geometry, height):
    if sun_vector[2] <= 0:
        return None

    polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
    shadow_lines = []

    for poly in polygons:
        base_coords = list(poly.exterior.coords)
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

    union_lines = unary_union(shadow_lines)
    return union_lines.convex_hull

# 为每个建筑物生成阴影
default_height = 3
building_gdf['shadow'] = building_gdf.apply(
    lambda row: shadow_using_lines(
        row.geometry,
        row[height_column] if height_column else default_height
    ), axis=1
)

# 获取建筑物边界范围
building_bounds = building_gdf.total_bounds  # (minx, miny, maxx, maxy)

# 下载自行车道路并转换为建筑物的坐标系
G = ox.graph_from_bbox(
    north=building_bounds[3],  # maxy
    south=building_bounds[1],  # miny
    east=building_bounds[2],   # maxx
    west=building_bounds[0],   # minx
    network_type="bike"
)
# 转换自行车道路的图为 GeoDataFrame 并变换坐标系
gdf_edges = ox.graph_to_gdfs(G, nodes=False)  # 获取道路边
gdf_edges = gdf_edges.to_crs(epsg=6669)  # 转换为 EPSG:6669

# 动态路径规划
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

# 绘制路径
def plot_path():
    global start_point, end_point
    start_node = ox.nearest_nodes(G, start_point[0], start_point[1])
    end_node = ox.nearest_nodes(G, end_point[0], end_point[1])
    try:
        route = nx.shortest_path(G, start_node, end_node, weight="length")
        ox.plot_graph_route(
            G,
            route,
            route_linewidth=3,
            node_size=0,
            bgcolor="k",
            ax=plt.gca(),
            show=False,
            close=False
        )
        plt.title("路径规划结果")
        plt.draw()
    except nx.NetworkXNoPath:
        print("起点和终点之间没有可用路径。")

# 可视化建筑物、阴影和自行车道路
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制建筑物
building_gdf.plot(ax=ax, color="lightblue", edgecolor="black", label="建筑物")

# 绘制阴影
shadow_gdf = building_gdf.set_geometry("shadow").dropna(subset=["shadow"])
shadow_gdf.plot(ax=ax, color="gray", alpha=0.5, label="阴影")

# 绘制自行车道路
ox.plot_graph(G, ax=ax, show=False, close=False)

# 添加图例
legend_handles = [
    Patch(color="lightblue", label="建筑物"),
    Patch(color="gray", alpha=0.5, label="阴影"),
    Patch(color="blue", label="自行车道路"),
]
plt.legend(handles=legend_handles, loc="upper right")
plt.title("建筑物、阴影与自行车道路", fontsize=16)
plt.xlabel("经度")
plt.ylabel("纬度")
plt.grid(True)

# 添加鼠标点击事件
fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()
