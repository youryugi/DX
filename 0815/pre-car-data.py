import pandas as pd

# 输入文件路径
input_csv = r"ocartrafficdata.csv"   # ← 改成你的文件路径
output_csv = r"ecoron01.csv"  # 输出文件路径

# 读取 CSV（自动识别常见编码）
for enc in ["utf-8", "utf-8-sig", "cp932", "gb18030"]:
    try:
        df = pd.read_csv(input_csv, encoding=enc)
        break
    except Exception:
        df = None
if df is None:
    raise ValueError("无法读取CSV，请检查文件或编码格式")

# 不区分大小写找到 car_name 列
cols_lower = {c.lower(): c for c in df.columns}
car_col = cols_lower.get("car_name")
if not car_col:
    raise ValueError("找不到 car_name 列，请确认表头是否正确")

# 筛选 car_name == ecoron01
df_filtered = df[df[car_col] == "ecoron01"].copy()

# 保存
df_filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"已筛选出 {len(df_filtered)} 行数据，保存到：{output_csv}")
