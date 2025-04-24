import pandas as pd
import folium
from folium.plugins import HeatMap
import requests

# 读取Excel文件
xlsx_file = r'C:\Users\79152\Desktop\Semi\dx\elle.xlsx'  # 请将文件名替换为你的文件名
df = pd.read_excel(xlsx_file)

# 提取经度和纬度数据
latitude = df.columns[0]  # 假设第一列是纬度
longitude = df.columns[1]  # 假设第二列是经度
location_data = df[[latitude, longitude]].dropna()

# 获取 Itami 市的地理边界数据
try:
    response = requests.get('https://nominatim.openstreetmap.org/search.php?q=Itami,+Hyogo,+Japan&polygon_geojson=1&format=json')
    response.raise_for_status()  # 检查请求是否成功
    itami_boundary = response.json()[0]['geojson']
except (requests.exceptions.RequestException, IndexError, KeyError, ValueError) as e:
    print(f"Error fetching Itami boundary data: {e}")
    itami_boundary = None

# 创建基础地图，使用OpenStreetMap
itami_map = folium.Map(location=[34.7855, 135.4017], zoom_start=13, tiles='OpenStreetMap')  # Itami市的中心位置

# 如果成功获取边界数据，则添加 Itami 市边界
if itami_boundary:
    folium.GeoJson(itami_boundary, name='Itami Boundary').add_to(itami_map)

# 添加热力图，调整点的半径和颜色渐变
heat_data = [[row[latitude], row[longitude]] for index, row in location_data.iterrows()]
HeatMap(heat_data, radius=10, blur=10, max_zoom=1).add_to(itami_map)

# 如果成功获取边界数据，则调整地图边界以适应 Itami 市
if itami_boundary:
    itami_map.fit_bounds(itami_map.get_bounds())

# 保存地图到HTML文件
output_file = 'itami_heatmap-eele.html'
itami_map.save(output_file)

print(f"热力图已保存到 '{output_file}'")
