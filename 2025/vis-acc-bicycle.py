import pandas as pd
import folium

# 1. 读取 Excel 表格
filepath = 'acc.xlsx'  # ← 替换为你的文件路径
df = pd.read_excel(filepath)

# 2. 提取经纬度（假设第一列是纬度，第二列是经度）
latitudes = df.iloc[:, 0].values
longitudes = df.iloc[:, 1].values

# 3. 初始化以神户市为中心的地图
kobe_center = [34.6901, 135.1955]  # 神户市中心点
m = folium.Map(location=kobe_center, zoom_start=13)

# 4. 在地图上添加点标记
for lat, lon in zip(latitudes, longitudes):
    folium.CircleMarker(location=[lat, lon],
                        radius=4,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7).add_to(m)

# 5. 保存为 HTML 地图文件
m.save('kobe_map_points.html')
print("✅ 地图已保存为 kobe_map_points.html")
