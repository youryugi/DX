import geopandas as gpd
import matplotlib.pyplot as plt

# 1. 读取 GML 文件
# 替换 'your_file.gml' 为你的 GML 文件路径
gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357465_bldg_6697_op.gml"
gdf = gpd.read_file(gml_file)

# 打印数据基本信息
print(gdf.head())
print(gdf.crs)  # 检查坐标系

# 2. 可视化数据
# 可视化 GML 中的所有几何对象
gdf.plot(figsize=(10, 10), color='blue', edgecolor='black')

# 添加标题
plt.title("GML Data Visualization")
plt.show()

# 3. 可视化某个字段（例如高度）
if 'measuredHeight' in gdf.columns:
    gdf.plot(column='measuredHeight', legend=True, cmap='viridis', figsize=(10, 10))
    plt.title("Height Distribution")
    plt.show()
else:
    print("No 'height' column found in the GML file.")
