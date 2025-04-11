import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta

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
    # 将高度列中的 null 值替换为默认高度 3 米
    building_gdf[height_column] = building_gdf[height_column].fillna(3)
    print(f"使用高度列 '{height_column}'，并将缺失值替换为 3 米。")
else:
    print("没有找到高度列，将在后续处理中使用默认高度 3 米。")


if not height_column:
    print("没有找到高度列，将从几何体中提取高度。")

# 确保坐标系一致，使用适当的投影（例如 EPSG:6669）
if building_gdf.crs.to_epsg() != 6669:
    print("Reprojecting building data to EPSG:6669 for accurate shadow calculation...")
    building_gdf = building_gdf.to_crs(epsg=6669)

# 设置大阪的位置
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)

# 设置日期和时间
date_time = datetime(2024, 12, 1, 16, 10, tzinfo=timezone(timedelta(hours=9)))

# 计算太阳高度角和方位角
solar_elevation = elevation(city.observer, date_time)
solar_azimuth = azimuth(city.observer, date_time)

print(f"太阳高度角: {solar_elevation:.2f}°")
print(f"太阳方位角: {solar_azimuth:.2f}°")
if solar_elevation <= 0:
    print("太阳位于地平线以下，无法生成阴影。")
    exit()

# 4. 修正太阳向量计算
sun_vector = np.array([
    np.cos(np.radians(solar_elevation)) * np.sin(np.radians(solar_azimuth)),
    np.cos(np.radians(solar_elevation)) * np.cos(np.radians(solar_azimuth)),
    np.sin(np.radians(solar_elevation))
])

# 5. 从三维坐标中提取建筑物的实际高度
def extract_building_height(geometry, default_height=3):
    if geometry.has_z:
        if geometry.geom_type == 'Polygon':
            z_values = [coord[2] for coord in geometry.exterior.coords if len(coord) > 2]
        elif geometry.geom_type == 'MultiPolygon':
            z_values = []
            for poly in geometry.geoms:
                z_values.extend([coord[2] for coord in poly.exterior.coords if len(coord) > 2])
        else:
            return default_height
        return max(z_values) if z_values else default_height
    else:
        return default_height

# 6. 根据太阳角度和高度生成阴影
def generate_shadow_3d(geometry, height):
    if sun_vector[2] <= 0:
        print("太阳在地平线以下，不生成阴影")
        return None

    if geometry.geom_type in ['Polygon', 'MultiPolygon']:
        # 将 MultiPolygon 和 Polygon 统一处理
        if geometry.geom_type == 'MultiPolygon':
            polygons = geometry.geoms
        else:
            polygons = [geometry]

        shadow_polygons = []
        for poly in polygons:
            if not poly.is_valid:
                poly = poly.buffer(0)

            # 提取建筑物底部的二维坐标
            base_coords = [(x, y) for x, y, *rest in poly.exterior.coords]
            print('base_coords',base_coords)
            if len(base_coords) < 3:
                continue  # 非法多边形，跳过

            # 计算阴影长度
            min_elevation_angle = 1e-3  # 防止除以零
            effective_elevation = max(solar_elevation, min_elevation_angle)
            shadow_length = height / np.tan(np.radians(effective_elevation))
            max_shadow_length = height * 100  # 限制阴影最大长度
            shadow_length = min(shadow_length, max_shadow_length)

            print(f"建筑物高度: {height} 米, 太阳高度角: {solar_elevation:.2f}°, 阴影长度: {shadow_length:.2f} 米")

            # 计算阴影顶点的二维坐标
            shadow_coords = []
            for x, y in base_coords:
                shadow_x = x - shadow_length * sun_vector[0]
                shadow_y = y - shadow_length * sun_vector[1]
                shadow_coords.append((shadow_x, shadow_y))
            print('shadow',shadow_coords)
            # 将建筑物底部坐标和阴影顶点连接起来
            full_coords = base_coords + shadow_coords[::-1]
            print('allshadow',full_coords)
            # 确保多边形闭合
            if full_coords[0] != full_coords[-1]:
                full_coords.append(full_coords[0])

            try:
                shadow_polygon = Polygon(full_coords)
                if not shadow_polygon.is_valid:
                    shadow_polygon = shadow_polygon.buffer(0)
                shadow_polygons.append(shadow_polygon)
            except Exception as e:
                print(f"创建多边形失败: {e}")
                continue

        if not shadow_polygons:
            print("未生成任何有效阴影")
            return None

        # 返回阴影的几何对象
        print('shadow_polygons', shadow_polygons)

        return MultiPolygon(shadow_polygons) if len(shadow_polygons) > 1 else shadow_polygons[0]

    else:
        print(f"不支持的几何类型: {geometry.geom_type}")
        return None

# 7. 为每栋建筑计算阴影
default_height = 3
building_gdf['shadow'] = building_gdf.apply(
    lambda row: generate_shadow_3d(
        row.geometry,
        extract_building_height(row.geometry, default_height) if height_column is None else row[height_column]
    ), axis=1
)

if 'shadow' not in building_gdf.columns or building_gdf['shadow'].isnull().all():
    print("未生成任何有效阴影。请检查数据或代码逻辑。")
    exit()

shadow_gdf = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')

# 8. 可视化
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

building_gdf.plot(ax=ax, color='lightblue', edgecolor='black', label='建筑物')
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='阴影')

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
