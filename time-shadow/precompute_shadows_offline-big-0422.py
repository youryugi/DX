import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import LineString
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timedelta, timezone
import pickle

interval_minutes=5
#######################################
# 函数1：计算单个时间点的合并阴影
#######################################
def compute_shadow_union_at_time(building_gdf, date_time, height_column='default_height'):
    city = LocationInfo(
        name="Osaka",
        region="Japan",
        timezone="Asia/Tokyo",
        latitude=34.6937,
        longitude=135.5023
    )
    solar_elev = elevation(city.observer, date_time)
    solar_azi = azimuth(city.observer, date_time)

    if solar_elev <= 0:
        return None  # 没太阳

    sun_vec = np.array([
        np.cos(np.radians(solar_elev)) * np.sin(np.radians(solar_azi)),
        np.cos(np.radians(solar_elev)) * np.cos(np.radians(solar_azi)),
        np.sin(np.radians(solar_elev))
    ])

    def shadow_for_building(geom, h):
        polygons = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        shadow_lines = []
        for poly in polygons:
            # 只取xy, 丢掉z
            base_coords = [(x, y) for x, y, *rest in poly.exterior.coords]

            shadow_coords = [
                (
                    x - h / np.tan(np.radians(solar_elev)) * sun_vec[0],
                    y - h / np.tan(np.radians(solar_elev)) * sun_vec[1]
                )
                for (x, y) in base_coords
            ]

            for base, shadow in zip(base_coords, shadow_coords):
                shadow_lines.append(LineString([base, shadow]))
        union_lines = unary_union(shadow_lines)
        return union_lines.convex_hull

    building_gdf['shadow_temp'] = building_gdf.apply(
        lambda row: shadow_for_building(row.geometry, row[height_column]),
        axis=1
    )

    shadow_gdf = building_gdf.dropna(subset=['shadow_temp']).set_geometry('shadow_temp')

    if shadow_gdf.empty:
        return None
    return unary_union(shadow_gdf.geometry)


#######################################
# 函数2：批量计算所有时刻的阴影
#######################################
def precompute_shadow_unions(building_gdf, start_dt, end_dt, interval_minutes, height_column='default_height'):
    current_time = start_dt
    time_to_union = {}
    while current_time <= end_dt:
        print(f"正在计算: {current_time}...")
        union_geom = compute_shadow_union_at_time(building_gdf, current_time, height_column)
        time_to_union[current_time] = union_geom
        current_time += timedelta(minutes=interval_minutes)
    return time_to_union


#######################################
# 读取GML数据
#######################################

bldg_gml_files = [
    r"bldg\51357451_bldg_6697_op.gml",
    r"bldg\51357452_bldg_6697_op.gml",
    r"bldg\51357453_bldg_6697_op.gml",
    r"bldg\51357461_bldg_6697_op.gml",
    r"bldg\51357462_bldg_6697_op.gml",
    r"bldg\51357463_bldg_6697_op.gml",
    r"bldg\51357471_bldg_6697_op.gml",
    r"bldg\51357472_bldg_6697_op.gml",
    r"bldg\51357473_bldg_6697_op.gml"

]

bldg_gdf_list = [gpd.read_file(file) for file in bldg_gml_files]
bldg_merged_gdf = pd.concat(bldg_gdf_list, ignore_index=True)

if bldg_merged_gdf.crs.to_epsg() != 6669:
    bldg_merged_gdf = bldg_merged_gdf.to_crs(epsg=6669)

height_column = None
for col in bldg_merged_gdf.columns:
    if 'height' in col.lower():
        height_column = col
        break
if height_column is None:
    height_column = 'default_height'
    bldg_merged_gdf[height_column] = 3.0
else:
    bldg_merged_gdf[height_column] = bldg_merged_gdf[height_column].fillna(3)

#######################################
# 设置时间段
#######################################

start_dt = datetime(2024, 12, 5, 13, tzinfo=timezone(timedelta(hours=9)))
end_dt = datetime(2024, 12, 5, 14, tzinfo=timezone(timedelta(hours=9)))

#######################################
# 开始批量计算并存储
#######################################

print(f"[INFO] 阴影计算范围: {start_dt} ~ {end_dt}, 每 {interval_minutes} 分钟")
time_to_union = precompute_shadow_unions(bldg_merged_gdf, start_dt, end_dt, interval_minutes, height_column)

valid_count = sum(1 for v in time_to_union.values() if v)
print(f"[INFO] 阴影计算完成，有效时间片数量: {valid_count}")

#######################################
# 存储
#######################################
# 自动拼接文件名
# 获取边界框经纬度（单位：度）
bounds = bldg_merged_gdf.to_crs(epsg=4326).total_bounds  # 转换为WGS84经纬度
minx, miny, maxx, maxy = bounds  # 左下角 (minx, miny)，右上角 (maxx, maxy)

# 保留 4 位小数
minx_str = f"{minx:.4f}"
miny_str = f"{miny:.4f}"
maxx_str = f"{maxx:.4f}"
maxy_str = f"{maxy:.4f}"

# 拼接文件名
date_str = start_dt.strftime("%Y%m%d")
start_str = start_dt.strftime("%H%M")
end_str = end_dt.strftime("%H%M")
output_file = (
    f"shadows_{date_str}_{start_str}_{end_str}_{interval_minutes}min_"
    f"LL_{minx_str}_{miny_str}_UR_{maxx_str}_{maxy_str}.pkl"
)

# 保存文件
with open(output_file, 'wb') as f:
    pickle.dump(time_to_union, f)

print(f"[SUCCESS] 阴影缓存已保存至 {output_file}")

