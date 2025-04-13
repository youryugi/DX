import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
import numpy as np

# 1. 读取 GML 文件
gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357465_bldg_6697_op.gml"
gdf = gpd.read_file(gml_file)

# 打印数据基本信息
print(gdf.head())
print(gdf.crs)  # 检查坐标系

# 确保坐标系为平面投影（米单位）
if gdf.crs.is_geographic:
    print("Transforming CRS to projected system...")
    gdf = gdf.to_crs(epsg=3857)  # 使用 EPSG:3857 或其他合适的平面投影

# 2. 定义太阳的方位角和高度角
solar_azimuth = 135  # 方位角（以北为0°顺时针）
solar_altitude = 45  # 高度角

# 3. 计算阴影长度比例
shadow_length_ratio = 1 / np.tan(np.radians(solar_altitude))

# 4. 检查高度列并计算阴影
if 'measuredHeight' not in gdf.columns:
    print("No 'measuredHeight' column found in the GML file.")
else:
    shadow_geometries = []
    for idx, row in gdf.iterrows():
        # 确保建筑物有高度，并且几何为 Polygon
        if row['measuredHeight'] > 0 and isinstance(row.geometry, Polygon):
            # 获取建筑物边界顶点
            exterior_coords = list(row.geometry.exterior.coords)
            height = row['measuredHeight']
            shadow_length = height * shadow_length_ratio

            # 计算阴影方向
            angle_rad = np.radians(solar_azimuth)
            shadow_dx = shadow_length * np.cos(angle_rad)
            shadow_dy = shadow_length * np.sin(angle_rad)

            # 计算阴影的顶点
            shadow_coords = [
                (x + shadow_dx, y + shadow_dy) for x, y in exterior_coords
            ]

            # 创建阴影的多边形
            shadow_polygon = Polygon(shadow_coords)
            shadow_geometries.append(shadow_polygon)
        else:
            shadow_geometries.append(None)

    # 创建新的 GeoDataFrame 并过滤无效几何
    shadow_gdf = gpd.GeoDataFrame(geometry=shadow_geometries, crs=gdf.crs)
    shadow_gdf = shadow_gdf[shadow_gdf.geometry.notnull() & shadow_gdf.is_valid]

    # 5. 可视化原始建筑和阴影
    ax = gdf.plot(figsize=(10, 10), color='blue', edgecolor='black', alpha=0.5)
    shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='Shadows')  # 使用灰色填充
    ax.set_aspect('equal')  # 设置默认纵横比
    plt.title("Building Shadows Visualization")
    plt.legend()
    plt.show()
