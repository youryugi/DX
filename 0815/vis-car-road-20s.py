# -*- coding: utf-8 -*-
"""
读取 CSV（只要 car_name == ecoron01），按真实时间顺序将点在底图上逐个“点亮”，
并在 20 秒内播完（无论点多少）。输出静态图和 GIF/MP4 动画。
"""
import sys
from pathlib import Path
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# ==== 配置 ====
CSV_PATH = r"ocartrafficdata.csv"   # ← 改成你的CSV路径或用命令行传参
TARGET_NAME = "ecoron01"
OUTPUT_PNG = "map_points.png"
OUTPUT_GIF = "map_points_animate.gif"
OUTPUT_MP4 = "map_points_animate.mp4"
POINT_SIZE = 14
TOTAL_DURATION_SEC = 20  # 动画总时长
# ============

def robust_read_csv(path):
    print("[1/5] 开始读取 CSV ...")
    t0 = time.perf_counter()
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "cp932", "gb18030"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            dt = time.perf_counter() - t0
            print(f"[1/5] 读取完成，用时 {dt:.2f}s，编码={enc}，行数={len(df)}，列数={len(df.columns)}")
            return df
        except Exception as e:
            last_err = e
    # 最后一次尝试默认编码
    try:
        df = pd.read_csv(path)
        dt = time.perf_counter() - t0
        print(f"[1/5] 读取完成（默认编码），用时 {dt:.2f}s，行数={len(df)}，列数={len(df.columns)}")
        return df
    except Exception:
        print(f"[1/5] 读取失败，最后错误：{last_err}")
        raise

def compute_real_time(df):
    """返回按真实时间排序后的 df，并新增 real_time（带时区信息用于排序/显示）"""
    print("[2/5] 开始计算真实时间戳(real_time) ...")
    t0 = time.perf_counter()

    cols = {c.lower(): c for c in df.columns}
    ms_col = cols.get("measurement_ms")
    dt_col = cols.get("measurement_datetime")

    if ms_col is None:
        print("[2/5] 未找到 measurement_ms，改用 measurement_datetime 排序。")
        if dt_col is None:
            raise ValueError("缺少 measurement_ms 和 measurement_datetime，无法排序")
        t1 = time.perf_counter()
        real_time = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.assign(real_time=real_time).sort_values("real_time").reset_index(drop=True)
        print(f"[2/5] 仅按 measurement_datetime 转换完成，用时 {time.perf_counter()-t1:.2f}s")
        print(f"[2/5] 总用时 {time.perf_counter()-t0:.2f}s")
        return df

    print("[2/5] 检查 measurement_ms 并判断是否为纪元毫秒 ...")
    t1 = time.perf_counter()
    df[ms_col] = pd.to_numeric(df[ms_col], errors="coerce")
    is_epoch_ms = df[ms_col].median() > 1e12
    print(f"[2/5] is_epoch_ms={is_epoch_ms}，检查用时 {time.perf_counter()-t1:.2f}s")

    if is_epoch_ms:
        print("[2/5] 纪元毫秒 → 转 UTC → 转 JST ...")
        t2 = time.perf_counter()
        df["real_time_utc"] = pd.to_datetime(df[ms_col], unit="ms", utc=True)
        t3 = time.perf_counter()
        df["real_time_jst"] = df["real_time_utc"].dt.tz_convert("Asia/Tokyo")
        print(f"[2/5] UTC 转换用时 {t3-t2:.2f}s，JST 转换用时 {time.perf_counter()-t3:.2f}s")
        df = df.assign(real_time=df["real_time_jst"])
    else:
        if dt_col is None:
            raise ValueError("measurement_ms 看起来是相对毫秒，但缺少 measurement_datetime 作为起点")
        print("[2/5] 相对毫秒 → measurement_datetime 起点 + 偏移 ...")
        t2 = time.perf_counter()
        start = pd.to_datetime(df.loc[0, dt_col], errors="coerce").tz_localize("Asia/Tokyo")
        ms_offset = df[ms_col] - df[ms_col].iloc[0]
        df["real_time_jst"] = start + pd.to_timedelta(ms_offset, unit="ms")
        df["real_time_utc"] = df["real_time_jst"].dt.tz_convert("UTC")
        print(f"[2/5] 相对时间计算用时 {time.perf_counter()-t2:.2f}s")
        df = df.assign(real_time=df["real_time_jst"])

    t4 = time.perf_counter()
    df = df.sort_values("real_time").reset_index(drop=True)
    print(f"[2/5] 排序用时 {time.perf_counter()-t4:.2f}s")
    print(f"[2/5] 真实时间戳计算总用时 {time.perf_counter()-t0:.2f}s")
    # 示例打印
    try:
        print("[2/5] 示例(real_time)前3行：", df["real_time"].head(3).astype(str).tolist())
    except Exception:
        pass
    return df

def build_geodf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    print("[3/5] 构建 GeoDataFrame ...")
    t0 = time.perf_counter()
    cols_lower = {c.lower(): c for c in df.columns}
    lat_col = cols_lower.get("latitude")
    lon_col = cols_lower.get("longitude")
    car_col = cols_lower.get("car_name")
    if not lat_col or not lon_col:
        raise ValueError("找不到 latitude/longitude 列")
    if not car_col:
        raise ValueError("找不到 car_name 列")

    n0 = len(df)
    df = df[df[car_col] == TARGET_NAME].copy()
    print(f"[3/5] 车名筛选：{n0} → {len(df)}")
    if df.empty:
        raise ValueError(f"没有 car_name = {TARGET_NAME} 的数据")

    df["__lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["__lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    mask = df["__lat"].between(-90, 90) & df["__lon"].between(-180, 180)
    n1 = len(df)
    df = df.loc[mask].copy()
    print(f"[3/5] 经纬度有效性过滤：{n1} → {len(df)}")

    # 计算真实时间并排序
    df = compute_real_time(df)

    gdf = gpd.GeoDataFrame(
        df.reset_index(drop=True),
        geometry=[Point(lon, lat) for lon, lat in zip(df["__lon"], df["__lat"])],
        crs="EPSG:4326"
    )
    print(f"[3/5] GeoDataFrame 构建完成，用时 {time.perf_counter()-t0:.2f}s，点数：{len(gdf)}")
    return gdf

def plot_static_png(gdf, out_png: str, point_size=12):
    print("[4/5] 绘制静态底图 PNG ...")
    t0 = time.perf_counter()
    g3857 = gdf.to_crs(epsg=3857)
    xmin, ymin, xmax, ymax = g3857.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    pad_x = max(dx * 0.05, 100)
    pad_y = max(dy * 0.05, 100)

    fig, ax = plt.subplots(figsize=(10, 10))
    g3857.plot(ax=ax, markersize=point_size*0.6, alpha=0.25)
    # 显式限制缩放，不让 contextily 推断到 27
    print("[4/5] 下载底图瓦片（zoom=18），若网络较慢这里会等一会儿 ...")
    #ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=g3857.crs, zoom=18)
    print("jiazaiwancheng")
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_axis_off()
    ax.set_title(f"All Points for {TARGET_NAME}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[4/5] PNG 输出完成：{out_png}（用时 {time.perf_counter()-t0:.2f}s）")

def animate_points(gdf, out_gif: str, out_mp4: str, point_size=14, total_sec=20):
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter
    import contextily as ctx
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    print("[5/5] 开始生成动画（严格 20s） ...")
    t0 = time.perf_counter()

    g3857 = gdf.to_crs(epsg=3857)
    xmin, ymin, xmax, ymax = g3857.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    pad_x = max(dx * 0.05, 100)
    pad_y = max(dy * 0.05, 100)

    fig, ax = plt.subplots(figsize=(10, 10))
    print("[5/5] 加载底图（zoom=18） ... 如网络不佳，会在此处等待。")
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=g3857.crs, zoom=18)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_axis_off()

    xs = g3857.geometry.x.values
    ys = g3857.geometry.y.values
    times = g3857["real_time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3].values
    n = len(xs)
    if n == 0:
        print("[5/5] [WARN] 没有点，无法动画。")
        return

    # —— 严格 20 秒 ——
    fps = 30
    frames = fps * total_sec  # 600
    if n >= frames:
        idxs = np.linspace(0, n - 1, frames, dtype=int)
    else:
        pad = np.full(frames - n, n - 1, dtype=int)
        idxs = np.concatenate([np.arange(n, dtype=int), pad])

    print(f"[5/5] 准备动画：点数={n}，fps={fps}，frames={frames}（20s）")

    scatter = ax.scatter([], [], s=point_size, alpha=0.9)
    ax.set_title(f"Time-lapse ({TARGET_NAME})  •  0/{n}", fontsize=14)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter,

    def update(frame_i):
        end_idx = idxs[frame_i]
        off = np.column_stack((xs[:end_idx+1], ys[:end_idx+1]))
        scatter.set_offsets(off)
        ax.set_title(
            f"Time-lapse ({TARGET_NAME})  •  {min(end_idx+1, n)}/{n}  •  {times[min(end_idx, n-1)]}",
            fontsize=14
        )
        return scatter,

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=int(1000/fps), blit=True)

    # —— 进度条：GIF ——
    try:
        pbar = tqdm(total=frames, desc="Rendering GIF", unit="frame")
        def _gif_progress(i, n_):
            pbar.update(1)
        ani.save(out_gif, writer=PillowWriter(fps=fps), dpi=150, progress_callback=_gif_progress)
        pbar.close()
        print(f"[5/5] GIF 完成：{out_gif}（20 秒，fps={fps}, frames={frames}）")
    except Exception as e:
        print(f"[5/5] [ERR] 保存 GIF 失败：{e}")

    # —— 进度条：MP4（需要 ffmpeg） ——
    try:
        pbar2 = tqdm(total=frames, desc="Rendering MP4", unit="frame")
        def _mp4_progress(i, n_):
            pbar2.update(1)
        ani.save(out_mp4, writer="ffmpeg", dpi=150, fps=fps, progress_callback=_mp4_progress)
        pbar2.close()
        print(f"[5/5] MP4 完成：{out_mp4}（20 秒，fps={fps}, frames={frames}）")
    except Exception as e:
        print(f"[5/5] [INFO] 未生成 MP4（可能未安装 ffmpeg）：{e}")

    print(f"[5/5] 动画总用时 {time.perf_counter()-t0:.2f}s")
    plt.close(fig)

def main(csv_path):
    t_all = time.perf_counter()
    df = robust_read_csv(csv_path)
    gdf = build_geodf(df)
    print(f"[INFO] {TARGET_NAME} 有效点数：{len(gdf)}")
    plot_static_png(gdf, OUTPUT_PNG, point_size=POINT_SIZE)
    animate_points(gdf, OUTPUT_GIF, OUTPUT_MP4, point_size=POINT_SIZE, total_sec=TOTAL_DURATION_SEC)
    print(f"[DONE] 全流程完成，总用时 {time.perf_counter()-t_all:.2f}s")

if __name__ == "__main__":
    csv = CSV_PATH if len(sys.argv) == 1 else sys.argv[1]
    if not Path(csv).exists():
        raise FileNotFoundError(f"找不到文件：{csv}")
    main(csv)
