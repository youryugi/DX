# 导入必要的库
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import skew
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime
import pytz
import pandas as pd

# 1. 加载建筑物数据
building_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357465_bldg_6697_op.gml"
building_gdf = gpd.read_file(building_gml_file)

# 检查建筑物坐标系
print(f"Building CRS: {building_gdf.crs}")

# 2. 确定高度列
height_column = None
for col in building_gdf.columns:
    if 'height' in col.lower():
        height_column = col
        break

if height_column is not None:
    # 将高度列中的值转换为数值类型，并替换缺失值为默认高度 3 米
    building_gdf[height_column] = pd.to_numeric(building_gdf[height_column], errors='coerce')
    building_gdf[height_column] = building_gdf[height_column].fillna(3)
    print(f"使用高度列 '{height_column}'，并将缺失值替换为 3 米。")
else:
    print("没有找到高度列，将在后续处理中使用默认高度 3 米。")
    # 如果没有高度列，添加一个默认高度列
    building_gdf['height'] = 3
    height_column = 'height'

# 3. 确保坐标系一致，使用适当的投影（例如 EPSG:6669）
if building_gdf.crs.to_epsg() != 6669:
    print("Reprojecting building data to EPSG:6669 for accurate shadow calculation...")
    building_gdf = building_gdf.to_crs(epsg=6669)

# 4. 设置大阪的位置
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)

# 5. 设置日期和时间（当地时间）
local_timezone = pytz.timezone("Asia/Tokyo")
local_time = local_timezone.localize(datetime(2024, 12, 1, 14, 10))

# 将当地时间转换为 UTC 时间
utc_time = local_time.astimezone(pytz.utc)

# 6. 计算太阳高度角和方位角
solar_elevation = elevation(city.observer, utc_time)
solar_azimuth = azimuth(city.observer, utc_time)

print(f"太阳高度角: {solar_elevation:.2f}°")
print(f"太阳方位角: {solar_azimuth:.2f}°")

if solar_elevation <= 0:
    print("太阳位于地平线以下，无法生成阴影。")
    exit()

# 7. 计算倾斜角度
tan_elevation = np.tan(np.radians(solar_elevation))
skew_angle_x = -tan_elevation * np.sin(np.radians(solar_azimuth))
skew_angle_y = -tan_elevation * np.cos(np.radians(solar_azimuth))

# 将倾斜角度转换为度数
skew_angle_x_deg = np.degrees(skew_angle_x)
skew_angle_y_deg = np.degrees(skew_angle_y)

print(f"倾斜角度 X: {skew_angle_x_deg:.4f}°")
print(f"倾斜角度 Y: {skew_angle_y_deg:.4f}°")

# 8. 计算高度列的平均值
mean_height = building_gdf[height_column].mean()

# 9. 定义生成阴影的函数
def generate_shadow_by_skew(geometry, skew_x, skew_y):
    if geometry.is_empty:
        return geometry
    if geometry.geom_type == 'Polygon':
        shadow = skew(geometry, skew_x, skew_y, origin='centroid', use_radians=False)
        return shadow
    elif geometry.geom_type == 'MultiPolygon':
        shadows = [skew(poly, skew_x, skew_y, origin='centroid', use_radians=False) for poly in geometry.geoms]
        return MultiPolygon(shadows)
    else:
        print(f"不支持的几何类型: {geometry.geom_type}")
        return None

# 10. 为每个建筑物生成阴影
building_gdf['shadow'] = building_gdf.apply(
    lambda row: generate_shadow_by_skew(
        row.geometry,
        skew_angle_x_deg * (row[height_column] / mean_height),
        skew_angle_y_deg * (row[height_column] / mean_height)
    ), axis=1
)

# 11. 检查是否生成了有效的阴影数据
if 'shadow' not in building_gdf.columns or building_gdf['shadow'].isnull().all():
    print("未生成任何有效阴影。请检查数据或代码逻辑。")
    exit()

# 创建阴影 GeoDataFrame
shadow_gdf = building_gdf[['shadow']].copy()
shadow_gdf = shadow_gdf.set_geometry('shadow')

# 12. 可视化
plt.rcParams['font.family'] = 'SimHei'
fig, ax = plt.subplots(figsize=(12, 8))

# 计算建筑物和阴影的总范围
combined_bounds = building_gdf.total_bounds
shadow_bounds = shadow_gdf.total_bounds

# 合并范围
x_min = min(combined_bounds[0], shadow_bounds[0])
y_min = min(combined_bounds[1], shadow_bounds[1])
x_max = max(combined_bounds[2], shadow_bounds[2])
y_max = max(combined_bounds[3], shadow_bounds[3])

buffer = (x_max - x_min) * 0.1  # 添加 10% 的缓冲区
x_min -= buffer
y_min -= buffer
x_max += buffer
y_max += buffer

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# 绘制阴影
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='阴影')

# 绘制建筑物
building_gdf.plot(ax=ax, color='lightblue', edgecolor='black', label='建筑物')

legend_handles = [
    Patch(facecolor='lightblue', edgecolor='black', label='建筑物'),
    Patch(facecolor='gray', edgecolor='black', label='阴影'),
]

plt.title("建筑物和阴影可视化", fontsize=16)
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel("X 坐标 (米)")
plt.ylabel("Y 坐标 (米)")
plt.grid(True)
plt.show()
