import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# 1. 读取建筑物的 GML 文件
building_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357465_bldg_6697_op.gml"
building_gdf = gpd.read_file(building_gml_file)

# 打印建筑物数据基本信息
print("Buildings Data:")
print(building_gdf.head())
print(building_gdf.crs)  # 检查坐标系

# 2. 读取道路的 GML 文件
road_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\tran\51357465_tran_6697_op.gml"
road_gdf = gpd.read_file(road_gml_file)

# 打印道路数据基本信息
print("Roads Data:")
print(road_gdf.head())
print(road_gdf.crs)  # 检查坐标系

# 3. 确保坐标系一致
if road_gdf.crs != building_gdf.crs:
    print("Reprojecting road data to match building CRS...")
    road_gdf = road_gdf.to_crs(building_gdf.crs)

# 4. 检查建筑物是否有高度字段
height_column = None
for col in building_gdf.columns:
    if 'height' in col.lower():
        height_column = col
        break

if height_column:
    print(f"Using '{height_column}' as the height column for visualization.")
else:
    print("No height column found. Defaulting to generic visualization.")
    height_column = None

# 5. 计算太阳阴影
# 设置太阳角度
solar_elevation = 30  # 太阳高度角 (degrees)
solar_azimuth = 135  # 太阳方位角 (degrees)

# 投影方向
dx = np.cos(np.radians(solar_azimuth)) / np.tan(np.radians(solar_elevation))
dy = np.sin(np.radians(solar_azimuth)) / np.tan(np.radians(solar_elevation))

# 生成阴影多边形
def generate_shadow(geometry, height):
    if geometry.geom_type == 'Polygon':
        shadow_length = height / np.tan(np.radians(solar_elevation))
        shadow_polygon = Polygon([
            (x + dx * shadow_length, y + dy * shadow_length)
            for x, y in geometry.exterior.coords
        ])
        return shadow_polygon
    return None

if height_column:
    building_gdf['shadow'] = building_gdf.apply(
        lambda row: generate_shadow(row.geometry, row[height_column]), axis=1
    )
else:
    print("No height column available for shadow calculation.")

# 6. 可视化建筑物、道路和阴影
fig, ax = plt.subplots(figsize=(12, 12))

# 绘制建筑物
if height_column:
    building_gdf.plot(
        column=height_column,
        cmap='viridis',
        legend=True,
        alpha=0.7,
        ax=ax,
        edgecolor='black',
        legend_kwds={'label': "Building Height (m)"}
    )
else:
    building_gdf.plot(
        color='blue',
        alpha=0.7,
        ax=ax,
        edgecolor='black',
        label="Buildings"
    )

# 绘制道路
road_gdf.plot(
    ax=ax,
    color='gray',
    linewidth=1.5,
    label="Roads"
)

# 绘制阴影
if height_column:
    shadow_gdf = building_gdf.set_geometry('shadow').dropna(subset=['shadow'])
    shadow_gdf.plot(
        color='black',
        alpha=0.5,
        ax=ax,
        label="Shadows"
    )

plt.title("Buildings, Roads, and Shadows")
plt.legend()
plt.show()
