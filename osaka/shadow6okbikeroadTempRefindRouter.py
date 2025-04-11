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
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from shapely.affinity import translate

# 假设上面的代码都已执行完毕，并且已经有:
# - fig, ax: 画布与坐标轴
# - gdf_edges: 包含路网信息的GeoDataFrame (已在EPSG:6669投影)
# - G: OSMnx生成的图
# - shadow_union: 合并后的阴影多边形
# - orig_node, dest_node: 起终点
shadow_union = unary_union(shadow_gdf.geometry)

shadow_lengths = []
# 定义一个函数，根据给定的coef重计算cool_weight和最凉爽路径
def update_cool_route(coef):
    # 根据coef重新计算cool_weight
    shadow_lengths = []
    for idx, row in gdf_edges.iterrows():
        edge_geom = row.geometry
        edge_length = edge_geom.length
        intersection_geom = edge_geom.intersection(shadow_union)
        shadowed_length = intersection_geom.length if not intersection_geom.is_empty else 0
        shadow_ratio = shadowed_length / edge_length if edge_length > 0 else 0

        # 使用新的系数coef影响阴影比：cost = length * (1 - coef * shadow_ratio)
        cost = edge_length * (1 - coef * shadow_ratio)
        shadow_lengths.append(cost)

    gdf_edges['cool_weight'] = shadow_lengths

    # 更新G中的cool_weight
    for (u,v,k), val in zip(gdf_edges.index, gdf_edges['cool_weight']):
        if (u,v,k) in G.edges:
            G[u][v][k]['cool_weight'] = val

    # 计算最凉爽路线
    new_cool_route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='cool_weight')
    new_cool_route_edges = [(new_cool_route[i], new_cool_route[i+1], 0) for i in range(len(new_cool_route)-1)]
    new_cool_route_gdf = gdf_edges.loc[new_cool_route_edges]

    # 为每个几何体应用一个小偏移使得与最短路径分开显示
    new_cool_route_gdf = new_cool_route_gdf.copy()
    new_cool_route_gdf['geometry'] = new_cool_route_gdf.geometry.apply(lambda g: translate(g, xoff=1.5, yoff=1.5))
    return new_cool_route_gdf

# 初始coef
initial_coef = 1
cool_route_gdf_updated = update_cool_route(initial_coef)
cool_route_line = cool_route_gdf_updated.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.legend()

# ---------------------
# 添加滑块和按钮
# ---------------------
# 为了在同一窗口添加slider和button，我们需要调整figure布局
plt.subplots_adjust(left=0.1, bottom=0.25)  # 给底部留出空间放置滑块和按钮

# 滑块位置和尺寸 [left, bottom, width, height]
ax_coef = plt.axes([0.1, 0.1, 0.65, 0.03])
coef_slider = Slider(
    ax=ax_coef,
    label='Shadow weight',
    valmin=-1.0,
    valmax=1.0,
    valinit=initial_coef,
    valstep=0.1
)
# 添加说明文字
ax_coef.text(
    0.5, -1.2,  # 文本位置 (slider 坐标轴内的位置，单位是归一化的坐标)
    'Weight is from -1 (ride under SUN) to 1 (ride under SHADOW). ',  # 文本内容
    ha='center',  # 水平对齐
    va='center',  # 垂直对齐
    transform=ax_coef.transAxes  # 使用 slider 坐标系
)

# 按钮位置和尺寸
ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Update Route')


def update_route(event):
    # 当按钮点击时，从slider获取当前coef值
    coef_val = coef_slider.val
    new_route_gdf = update_cool_route(coef_val)

    # 清除之前的最凉爽路线图层（若需要）
    # 简单的方法是重新绘制整张图，但可以选择性地移除旧路线图层，这里简单处理：
    # 我们先移除上一次绘制的图形对象，然后再绘制新的路线
    # 这里cool_route_line是一个GeoSeriesPlot生成的对象可能是一个集合，需要简单处理
    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    # 绘制新路线
    new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
    plt.draw()


button.on_clicked(update_route)

plt.show()
