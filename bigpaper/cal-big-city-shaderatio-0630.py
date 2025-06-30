import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta
import osmnx as ox
import pandas as pd
from shapely.ops import unary_union
import os

# ==== 参数设置 ====
bldg_dir = os.path.join('..', 'time-shadow', 'bldg')
road_dir = os.path.join('..', 'time-shadow', 'tran')
bldg_gml_files = [os.path.join(bldg_dir, f) for f in [
    "51357451_bldg_6697_op.gml",
    "51357452_bldg_6697_op.gml",
    "51357453_bldg_6697_op.gml",
    "51357461_bldg_6697_op.gml",
    "51357462_bldg_6697_op.gml",
    "51357463_bldg_6697_op.gml",
    "51357471_bldg_6697_op.gml",
    "51357472_bldg_6697_op.gml",
    "51357473_bldg_6697_op.gml"
]]
road_gml_files = [os.path.join(road_dir, f) for f in [
    "51357451_tran_6697_op.gml",
    "51357452_tran_6697_op.gml",
    "51357453_tran_6697_op.gml",
    "51357461_tran_6697_op.gml",
    "51357462_tran_6697_op.gml",
    "51357463_tran_6697_op.gml",
    "51357471_tran_6697_op.gml",
    "51357472_tran_6697_op.gml",
    "51357473_tran_6697_op.gml"
]]

# ==== 时间参数 ====
start_hour = 9
end_hour = 10
step_min = 30
date_str = "2025-07-05"
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)

# ==== 读取数据 ====
bldg_gdf_list = [gpd.read_file(file) for file in bldg_gml_files]
building_gdf = gpd.GeoDataFrame(pd.concat(bldg_gdf_list, ignore_index=True))
road_gdf_list = [gpd.read_file(file) for file in road_gml_files]
road_gdf = gpd.GeoDataFrame(pd.concat(road_gdf_list, ignore_index=True))

# ==== 投影 ====
if building_gdf.crs.to_epsg() != 6669:
    building_gdf = building_gdf.to_crs(epsg=6669)
if road_gdf.crs.to_epsg() != 6669:
    road_gdf = road_gdf.to_crs(epsg=6669)

# ==== 高度列 ====
height_column = None
for col in building_gdf.columns:
    if 'height' in col.lower():
        height_column = col
        break
if height_column is not None:
    building_gdf[height_column] = building_gdf[height_column].fillna(3)
else:
    height_column = 'default_height'
    building_gdf[height_column] = 3.0

# ==== 阴影函数 ====
def shadow_using_lines(geometry, height, solar_elevation, sun_vector):
    if sun_vector[2] <= 0:
        return None
    polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
    shadow_lines = []
    for poly in polygons:
        base_coords = [(x, y) for x, y, *rest in poly.exterior.coords]
        shadow_coords = [
            (
                x - height / np.tan(np.radians(solar_elevation)) * sun_vector[0],
                y - height / np.tan(np.radians(solar_elevation)) * sun_vector[1]
            )
            for x, y in base_coords
        ]
        for base, shadow in zip(base_coords, shadow_coords):
            shadow_lines.append(LineString([base, shadow]))
    union_lines = unary_union(shadow_lines)
    shadow_polygon = union_lines.convex_hull
    return shadow_polygon

# ==== 生成时间点 ====
tz = timezone(timedelta(hours=9))
date_base = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=start_hour, minute=0, second=0, tzinfo=tz)
num_steps = int((end_hour - start_hour) * 60 / step_min) + 1
time_points = [date_base + timedelta(minutes=step_min * i) for i in range(num_steps)]

# ==== 计算阴影比 ====
result = []
for dt in time_points:
    solar_elevation = elevation(city.observer, dt)
    solar_azimuth = azimuth(city.observer, dt)
    if solar_elevation <= 0:
        shade_ratio = 0
        print(f"{dt.strftime('%H:%M')}: 太阳高度角<=0，阴影比为0")
        result.append({'time': dt.strftime('%H:%M'), 'shade_ratio': 0})
        continue
    sun_vector = np.array([
        np.cos(np.radians(solar_elevation)) * np.sin(np.radians(solar_azimuth)),
        np.cos(np.radians(solar_elevation)) * np.cos(np.radians(solar_azimuth)),
        np.sin(np.radians(solar_elevation))
    ])
    building_gdf['shadow'] = building_gdf.apply(
        lambda row: shadow_using_lines(row.geometry, row[height_column], solar_elevation, sun_vector), axis=1
    )
    shadow_gdf = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')
    shadow_union = unary_union(shadow_gdf.geometry)
    total_length = 0
    shadowed_length = 0
    for idx, row in road_gdf.iterrows():
        edge_geom = row.geometry
        edge_length = edge_geom.length
        intersection_geom = edge_geom.intersection(shadow_union)
        shadow_len = intersection_geom.length if not intersection_geom.is_empty else 0
        total_length += edge_length
        shadowed_length += shadow_len
    shade_ratio = shadowed_length / total_length if total_length > 0 else 0
    print(f"{dt.strftime('%H:%M')}: 阴影比={shade_ratio:.4f}")
    result.append({'time': dt.strftime('%H:%M'), 'shade_ratio': shade_ratio})

# ==== 保存结果 ====
df_result = pd.DataFrame(result)
df_result.to_csv('shade_ratio_{}_{}-{}_step{}.csv'.format(
    date_str, start_hour, end_hour, step_min), index=False, encoding='utf-8-sig')
print("已保存为CSV。")
