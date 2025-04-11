import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# 1. 加载建筑物数据
building_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357465_bldg_6697_op.gml"
building_gdf = gpd.read_file(building_gml_file)

# 2. 确定高度列
height_column = None
for col in building_gdf.columns:
    if 'height' in col.lower():
        height_column = col
        break

if not height_column:
    print("没有找到高度列。程序终止。")
    exit()

# 3. 设置太阳角度
solar_elevation = 30  # 太阳高度角 (度)
solar_azimuth = 135  # 太阳方位角 (度)

# 4. 计算太阳方向向量
sun_vector = np.array([
    np.cos(np.radians(solar_azimuth)) * np.cos(np.radians(solar_elevation)),
    np.sin(np.radians(solar_azimuth)) * np.cos(np.radians(solar_elevation)),
    np.sin(np.radians(solar_elevation))
])


# 5. 从三维坐标中提取建筑物的实际高度
def extract_building_height(geometry, default_height):
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
    """生成建筑阴影的多边形，基于三维几何和高度。"""
    if geometry.geom_type in ['Polygon', 'MultiPolygon']:
        polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]

        shadow_polygons = []
        for poly in polygons:
            if not poly.has_z:
                continue

            # 提取多边形的最高高度
            actual_height = extract_building_height(poly, height)

            # 计算投影阴影
            top_coords = [(x, y, z) for x, y, z in poly.exterior.coords]
            shadow_coords = []
            for x, y, z in top_coords:
                shadow_length = actual_height / sun_vector[2]
                shadow_x = x + shadow_length * sun_vector[0]
                shadow_y = y + shadow_length * sun_vector[1]
                shadow_coords.append((shadow_x, shadow_y))

            shadow_polygon = Polygon(shadow_coords)
            shadow_polygons.append(shadow_polygon)

        return MultiPolygon(shadow_polygons) if len(shadow_polygons) > 1 else shadow_polygons[0]
    return None


# 7. 为每栋建筑计算阴影
default_height = 3  # 缺省高度
building_gdf['shadow'] = building_gdf.apply(
    lambda row: generate_shadow_3d(
        row.geometry,
        row[height_column] if row[height_column] is not None else default_height
    ), axis=1
)

# 8. 打印结果
for index, row in building_gdf.iterrows():
    print(f"建筑 {index}:")
    print(f"  几何体: {row['geometry']}")
    print(f"  高度: {row[height_column] if row[height_column] is not None else default_height}")
    print(f"  阴影: {row['shadow']}\n")

if 'shadow' not in building_gdf.columns:
    print("未生成阴影数据，请先运行阴影计算代码。")
    exit()

plt.rcParams['font.family'] = 'SimHei'

# 加载前面的建筑物数据和阴影生成代码
if 'shadow' not in building_gdf.columns:
    print("未生成阴影数据，请先运行阴影计算代码。")
    exit()

# 检查是否存在阴影数据
shadow_gdf = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')

# 创建图例的自定义内容
legend_handles = [
    Patch(facecolor='lightblue', edgecolor='black', label='建筑物'),
    Patch(facecolor='gray', edgecolor='black', label='阴影'),
]

# 绘图
fig, ax = plt.subplots(figsize=(12, 8))
# 计算建筑物和阴影的边界范围
bounds = building_gdf.total_bounds  # 返回 [xmin, ymin, xmax, ymax]

# 为范围添加适当的缓冲区
buffer = 0.0001  # 根据数据范围调整此值
x_min, y_min, x_max, y_max = bounds
x_min -= buffer
y_min -= buffer
x_max += buffer
y_max += buffer

# 设置绘图区域
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# 绘制建筑物
building_gdf.plot(ax=ax, color='lightblue', edgecolor='black')

# 绘制阴影
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5)

# 添加标题和图例
plt.title("建筑物和阴影可视化", fontsize=16)
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel("经度")
plt.ylabel("纬度")
plt.grid(True)

# 显示图形
plt.show()