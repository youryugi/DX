import pandas as pd

# 读取Excel文件
xlsx_file = 'output_with_way_ids_and_tags_final.xlsx'
df = pd.read_excel(xlsx_file)

# 假设前4列不是标签列（根据实际情况调整）
non_tag_columns = df.columns[:4]
tag_columns = df.drop(non_tag_columns, axis=1)

# 统计所有标签列中每个非NaN值的出现次数，同时记录每个值对应的标签列
value_tag_counts = tag_columns.stack().reset_index()
value_tag_counts.columns = ['row', 'tag', 'value']
value_tag_counts = value_tag_counts.dropna(subset=['value'])
value_counts = value_tag_counts.groupby(['value', 'tag']).size().reset_index(name='count')

# 找出出现次数最多的前20个值
top_20_values = value_counts.groupby('value')['count'].sum().nlargest(20).reset_index()

# 只保留前20个值的数据
top_20_value_details = value_counts[value_counts['value'].isin(top_20_values['value'])]

# 输出结果到Excel文件
output_file = 'top_20_values_tags_counts.xlsx'
top_20_value_details.to_excel(output_file, sheet_name='Top 20 Values Tags Counts', index=False)

print(f"Top 20 values and their counts with tags written to {output_file}")
