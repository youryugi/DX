import pandas as pd

# 读取原始数据
df = pd.read_csv("ecoron01.csv")

# 确保是数值类型
df["measurement_ms"] = pd.to_numeric(df["measurement_ms"], errors="coerce")

# 判断：看起来像绝对“纪元毫秒”的阈值（> 1e12 基本可认定是 ms since 1970）
is_epoch_ms = df["measurement_ms"].median() > 1e12

if is_epoch_ms:
    # 绝对纪元毫秒 → 直接转为带时区的时间戳
    df["real_time_utc"] = pd.to_datetime(df["measurement_ms"], unit="ms", utc=True)
    df["real_time_jst"] = df["real_time_utc"].dt.tz_convert("Asia/Tokyo")
else:
    # 退路：measurement_ms 只是相对偏移 → 用 measurement_datetime 作为起点
    # 如果你的起点就是日本时间，请本行保留 Asia/Tokyo；否则改成你的时区
    start_time = pd.to_datetime(df.loc[0, "measurement_datetime"], format="%Y/%m/%d %H:%M:%S")
    start_time = start_time.tz_localize("Asia/Tokyo")  # 若原始字符串没有时区信息
    # 相对首条的毫秒偏移
    ms_offset = df["measurement_ms"] - df["measurement_ms"].iloc[0]
    # 得到 JST 与 UTC
    df["real_time_jst"] = start_time + pd.to_timedelta(ms_offset, unit="ms")
    df["real_time_utc"] = df["real_time_jst"].dt.tz_convert("UTC")

# 便于查看：保留到毫秒的字符串列（不会丢毫秒）
df["real_time_jst_ms"] = df["real_time_jst"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
df["real_time_utc_ms"] = df["real_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]

# 保存新文件
df.to_csv("ecoron01_with_timestamp.csv", index=False)

print(df[["measurement_ms", "real_time_utc_ms", "real_time_jst_ms"]].head(10))
