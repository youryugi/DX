import pandas as pd

# 读取Excel文件
xlsx_file = r'C:\Users\79152\Desktop\Semi\dx\output_with_way_ids_and_tags_final.xlsx'  # 请将文件名替换为你的文件名
df = pd.read_excel(xlsx_file)

# 将第三列的空白部分填充为上面的单元格的值
third_column = df.columns[2]  # 假设第三列是目标列
df[third_column] = df[third_column].fillna(method='ffill')

# 保存处理后的数据到新的Excel文件
output_file = r'C:\Users\79152\Desktop\Semi\dx\itami2_filled.xlsx'
df.to_excel(output_file, index=False)

print(f"填充后的数据已保存到 '{output_file}'")
