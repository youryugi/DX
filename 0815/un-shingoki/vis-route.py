import os
import math
import pandas as pd
import folium
from folium import plugins

def _detect_lat_lon_columns(columns):
    cols_low = [c.lower() for c in columns]
    lat_candidates = ["latitude", "lat", "y", "lat_deg"]
    lon_candidates = ["longitude", "lon", "lng", "x", "long", "lon_deg"]
    lat_col = next((c for c in columns if c.lower() in lat_candidates), None)
    lon_col = next((c for c in columns if c.lower() in lon_candidates), None)
    return lat_col, lon_col

def _detect_time_column(columns):
    pref = ["real_measurement_time","timestamp","time","datetime",
            "measurement_datetime","measurement_ms","epoch","date"]
    for p in pref:
        for c in columns:
            if c.lower() == p:
                return c
    for c in columns:
        cl = c.lower()
        if "time" in cl or cl.endswith("_ms"):
            return c
    return None

def _clean_track(df, lat_col, lon_col):
    df = df.copy()
    df = df[pd.notna(df[lat_col]) & pd.notna(df[lon_col])]
    df = df[(df[lat_col].between(-90, 90)) & (df[lon_col].between(-180, 180))]
    df = df.loc[(df[lat_col].shift() != df[lat_col]) | (df[lon_col].shift() != df[lon_col])]
    return df

def _to_points(df, lat_col, lon_col):
    return list(zip(df[lat_col].astype(float), df[lon_col].astype(float)))

def _bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1)*math.cos(phi2) - math.sin(phi1)*math.sin(phi2)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def _add_direction_arrows(m, points, every=25):
    for i in range(0, len(points) - 1, every):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i+1]
        brg = _bearing(lat1, lon1, lat2, lon2)
        folium.RegularPolygonMarker(
            location=(lat2, lon2),
            number_of_sides=3,
            radius=6,
            rotation=brg,
            fill=True
        ).add_to(m)

def build_map_for_csv(csv_path, out_folder, tiles="OpenStreetMap"):
    os.makedirs(out_folder, exist_ok=True)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[跳过] 无法读取 {csv_path}: {e}")
        return None
    if df.empty:
        print(f"[跳过] 空文件: {csv_path}")
        return None

    lat_col, lon_col = _detect_lat_lon_columns(df.columns)
    if not lat_col or not lon_col:
        print(f"[跳过] 未找到经纬度列: {csv_path}")
        return None

    tcol = _detect_time_column(df.columns)
    if tcol is not None:
        try:
            if df[tcol].dtype.kind in "ifu":
                ser = df[tcol]
                if ser.abs().max() > 1e12:
                    dt = pd.to_datetime(ser, unit="ms", errors="coerce")
                elif ser.abs().max() > 1e10:
                    dt = pd.to_datetime(ser, unit="us", errors="coerce")
                else:
                    dt = pd.to_datetime(ser, unit="s", errors="coerce")
                df = df.assign(__dt=dt).sort_values("__dt").drop(columns="__dt")
            else:
                dt = pd.to_datetime(df[tcol], errors="coerce")
                df = df.assign(__dt=dt).sort_values("__dt").drop(columns="__dt")
        except Exception:
            pass

    df = _clean_track(df, lat_col, lon_col)
    if len(df) < 2:
        print(f"[跳过] 有效点少于2个: {csv_path}")
        return None

    points = _to_points(df, lat_col, lon_col)
    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=15, tiles=tiles)

    folium.PolyLine(points, weight=4, opacity=0.9).add_to(m)
    try:
        plugins.AntPath(points, delay=600, dash_array=[10, 20]).add_to(m)
    except Exception:
        pass

    folium.Marker(points[0], tooltip="Start", popup="Start", icon=folium.Icon(icon="play")).add_to(m)
    folium.Marker(points[-1], tooltip="End", popup="End", icon=folium.Icon(icon="flag")).add_to(m)
    _add_direction_arrows(m, points, every=max(1, len(points)//30))

    out_path = os.path.join(out_folder, f"{name}.html")
    m.save(out_path)
    print(f"[OK] {csv_path} → {out_path}")
    return out_path

def generate_maps_for_folder(src_folder, out_folder, tiles="OpenStreetMap"):
    results = []
    for root, _, files in os.walk(src_folder):
        for f in files:
            if f.lower().endswith(".csv"):
                p = os.path.join(root, f)
                out = build_map_for_csv(p, out_folder, tiles=tiles)
                if out:
                    results.append(out)
    print(f"完成，共生成 {len(results)} 个地图。")
    return results

if __name__ == "__main__":
    # Windows 示例：
    src = r"gpsdata-csv"      # 放你的 CSV 的文件夹
    dst = r"vis-route"  # 输出 HTML 地图的文件夹
    generate_maps_for_folder(src, dst, tiles="OpenStreetMap")
