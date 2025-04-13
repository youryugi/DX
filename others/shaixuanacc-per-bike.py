import pandas as pd

# 读取第一个Excel文件
file1 = r'C:\Users\79152\Desktop\Semi\dx\bike-person.xlsx'  # 请将文件名替换为你的文件名
df1 = pd.read_excel(file1)

# 读取第二个Excel文件
file2 = r'C:\Users\79152\Desktop\Semi\dx\itami2_filled.xlsx'  # 请将文件名替换为你的文件名
df2 = pd.read_excel(file2)

# 假设第十七列和第三列分别是标号列
label_col_df1 = df1.columns[16]  # 第一个表格的第十七列
label_col_df2 = df2.columns[2]   # 第二个表格的第三列

# 合并两个数据框，基于标号列
merged_df = df1.merge(df2, left_on=label_col_df1, right_on=label_col_df2, suffixes=('_df1', '_df2'))

# 输出合并后的数据框到新的Excel文件
output_file = r'C:\Users\79152\Desktop\Semi\dx\merged_output.xlsx'
merged_df.to_excel(output_file, index=False)

print(f"合并后的数据已保存到 '{output_file}'")
