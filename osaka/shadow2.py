import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

if not height_column:
    print("没有找到高度列，默认使用高度 3 米。")
    height_column = None

# 确保坐标系一致
if building_gdf.crs.to_epsg() != 3857:
    print("Reprojecting building data to EPSG:3857 for accurate shadow calculation...")
    building_gdf = building_gdf.to_crs(epsg=3857)

from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta

# 设置大阪的位置
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)

# 设置日期和时间
date_time = datetime(2024, 12, 1, 16, 0, tzinfo=timezone(timedelta(hours=9)))  # 示例时间：2024年12月1日12:00

# 计算太阳高度角和方位角
solar_elevation = elevation(city.observer, date_time)  # 太阳高度角
solar_azimuth = azimuth(city.observer, date_time)      # 太阳方位角

print(f"太阳高度角: {solar_elevation:.2f}°")
print(f"太阳方位角: {solar_azimuth:.2f}°")
if solar_elevation <= 0:
    print("太阳位于地平线以下，无法生成阴影。")
    exit()

# 4. 计算太阳方向向量
sun_vector = np.array([
    -np.cos(np.radians(solar_azimuth)) * np.cos(np.radians(solar_elevation)),
    -np.sin(np.radians(solar_azimuth)) * np.cos(np.radians(solar_elevation)),
    np.sin(np.radians(solar_elevation))
])


# 5. 从三维坐标中提取建筑物的实际高度
def extract_building_height(geometry, default_height=3):
    """从三维几何提取高度，若失败返回默认高度。"""
    if geometry.geom_type == 'Polygon':
        z_values = [z for x, y, z in geometry.exterior.coords]
    elif geometry.geom_type == 'MultiPolygon':
        z_values = []
        for poly in geometry.geoms:
            z_values.extend([z for x, y, z in poly.exterior.coords])
    else:
        return default_height

    # 返回最高点高度或默认高度
    return max(z_values) if z_values else default_height

# 6. 根据太阳角度和高度生成阴影
def generate_shadow_3d(geometry, height):
    """生成建筑阴影的二维多边形，基于建筑物高度和太阳方向。"""
    if sun_vector[2] <= 0:
        print("太阳在地平线以下，不生成阴影")
        return None

    if geometry.geom_type in ['Polygon', 'MultiPolygon']:
        polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]

        shadow_polygons = []
        for poly in polygons:
            # 修复无效几何
            if not poly.is_valid:
                print(f"修复无效几何: {poly}")
                poly = poly.buffer(0)

            # 提取多边形的二维坐标
            top_coords = [(x, y) for x, y, *_ in poly.exterior.coords]
            print(f"原始坐标: {top_coords}")

            shadow_coords = []
            shadow_length = height / np.tan(np.radians(solar_elevation))  # 阴影长度公式

            # 计算阴影的二维坐标
            for x, y in top_coords:
                shadow_x = x + shadow_length * sun_vector[0]
                shadow_y = y + shadow_length * sun_vector[1]
                shadow_coords.append((shadow_x, shadow_y))

            print(f"生成的阴影坐标: {shadow_coords}")

            # 确保多边形闭合
            if shadow_coords and shadow_coords[0] != shadow_coords[-1]:
                print(f"闭合多边形: 添加点 {shadow_coords[0]} 到末尾")
                shadow_coords.append(shadow_coords[0])

            # 检查是否有足够的点构成多边形
            if len(shadow_coords) < 4:
                print("生成的阴影点不足以构成多边形，跳过")
                continue

            # 创建二维阴影多边形
            try:
                shadow_polygon = Polygon(shadow_coords)
                shadow_polygons.append(shadow_polygon)
            except Exception as e:
                print(f"创建多边形失败: {e}")
                print(f"阴影坐标: {shadow_coords}")
                continue

        if not shadow_polygons:
            print("未生成任何有效阴影")
            return None

        return MultiPolygon(shadow_polygons) if len(shadow_polygons) > 1 else shadow_polygons[0]
    else:
        print(f"不支持的几何类型: {geometry.geom_type}")
        return None

# 7. 为每栋建筑计算阴影
default_height = 3  # 缺省高度
building_gdf['shadow'] = building_gdf.apply(
    lambda row: generate_shadow_3d(
        row.geometry,
        row[height_column] if height_column and row[height_column] is not None else default_height
    ), axis=1
)

# 检查是否生成了有效的阴影数据
if 'shadow' not in building_gdf.columns or building_gdf['shadow'].isnull().all():
    print("未生成任何有效阴影。请检查数据或代码逻辑。")
    exit()

# 提取有效阴影数据
shadow_gdf = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')

# 8. 可视化
plt.rcParams['font.family'] = 'SimHei'
fig, ax = plt.subplots(figsize=(12, 8))

# 计算建筑物和阴影的边界范围
bounds = building_gdf.total_bounds  # 返回 [xmin, ymin, xmax, ymax]

# 为范围添加适当的缓冲区
buffer = 10  # 根据坐标系单位调整（EPSG:3857 中单位为米）
x_min, y_min, x_max, y_max = bounds
x_min -= buffer
y_min -= buffer
x_max += buffer
y_max += buffer

# 设置绘图区域
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# 绘制建筑物
building_gdf.plot(ax=ax, color='lightblue', edgecolor='black', label='建筑物')

# 绘制阴影
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='阴影')

# 创建图例
legend_handles = [
    Patch(facecolor='lightblue', edgecolor='black', label='建筑物'),
    Patch(facecolor='gray', edgecolor='black', label='阴影'),
]

# 添加标题和图例
plt.title("建筑物和阴影可视化", fontsize=16)
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel("经度")
plt.ylabel("纬度")
plt.grid(True)

# 显示图形
plt.show()
