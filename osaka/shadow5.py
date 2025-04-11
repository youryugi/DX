import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta
# 加载建筑物数据
building_gml_file =  r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357465_bldg_6697_op.gml"
building_gdf = gpd.read_file(building_gml_file)
# 设置大阪的位置和时间
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)
date_time = datetime(2024, 12, 1, 16, 10, tzinfo=timezone(timedelta(hours=9)))

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
def precise_shadow_projection(geometry, height):
    if sun_vector[2] <= 0:
        print("太阳位于地平线以下，不生成阴影。")
        return None

    polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
    shadow_polygons = []

    for poly in polygons:
        base_coords = [(x, y) for x, y, *rest in poly.exterior.coords]
        if len(base_coords) < 3:
            continue  # 跳过无效多边形

        # 计算阴影长度，并限制最大值
        max_shadow_length = height * 10  # 限制阴影最大长度
        shadow_coords = [
            (
                x - min(max_shadow_length, height / np.tan(np.radians(solar_elevation))) * sun_vector[0],
                y - min(max_shadow_length, height / np.tan(np.radians(solar_elevation))) * sun_vector[1]
            )
            for x, y in base_coords
        ]

        # 检查连接点
        full_coords = base_coords + shadow_coords[::-1]
        if full_coords[0] != full_coords[-1]:
            full_coords.append(full_coords[0])  # 确保闭合

        # 创建阴影多边形
        try:
            shadow_polygon = Polygon(full_coords)
            if not shadow_polygon.is_valid:
                shadow_polygon = shadow_polygon.buffer(0)  # 修复无效几何
            shadow_polygons.append(shadow_polygon)
        except Exception as e:
            print(f"创建阴影多边形失败: {e}")

    return MultiPolygon(shadow_polygons) if len(shadow_polygons) > 1 else shadow_polygons[0]

# 为每个建筑物生成阴影
default_height = 3
building_gdf['shadow'] = building_gdf.apply(
    lambda row: precise_shadow_projection(
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
plt.show()
