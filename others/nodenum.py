import pandas as pd

# 读取用户上传的xlsx文件
xlsx_file = 'output_with_way_ids_and_tags_final.xlsx'
df = pd.read_excel(xlsx_file)

# 统计每个tag列的非空值出现次数
tag_counts = df.notna().sum()

# 去掉前几列非标签列（根据实际情况调整）
non_tag_columns = df.columns[:4]  # 假设前4列不是tag列
tag_counts = tag_counts.drop(non_tag_columns)

# 找出出现次数最多的前20个tag
top_20_tags = tag_counts.nlargest(20)
print("Top 20 tags and their counts:")
print(top_20_tags)

# 输出结果到Excel文件
output_file = 'top_20_tags_counts.xlsx'
top_20_tags.to_excel(output_file)
print(f"Top 20 tags and their counts written to {output_file}")
