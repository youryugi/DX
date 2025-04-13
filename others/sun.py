import math
from shapely.geometry import Polygon, LineString

# 示例输入
building_vertices = [
    (135.45477881799263, 34.73332920277088, 9.39),  # (x, y, z)
    (135.4546908032345, 34.733344179331425, 9.39),
    (135.45454912979577, 34.7333439056032, 9.39)
]
sun_azimuth = math.radians(45)  # 太阳方位角（度转弧度）
sun_elevation = math.radians(30)  # 太阳高程角（度转弧度）

# 计算影子投影向量
shadow_vector = (-math.cos(sun_azimuth) * math.tan(sun_elevation),
                 -math.sin(sun_azimuth) * math.tan(sun_elevation),
                 -1)

# 投影到地面
shadow_points = []
for x, y, z in building_vertices:
    shadow_x = x + shadow_vector[0] * z
    shadow_y = y + shadow_vector[1] * z
    shadow_points.append((shadow_x, shadow_y))

# 构建阴影边界
shadow_polygon = Polygon(shadow_points)

# 可视化
import matplotlib.pyplot as plt
x, y = shadow_polygon.exterior.xy
plt.plot(x, y, label="Shadow")
plt.scatter(*zip(*[(p[0], p[1]) for p in building_vertices]), color="red", label="Building")
plt.legend()
plt.show()
