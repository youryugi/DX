import pandas as pd

# 读取CSV文件
csv_file = 'output_with_tags.csv'
df = pd.read_csv(csv_file)

# 假设前4列不是标签列（根据实际情况调整）
non_tag_columns = df.columns[:1]
tag_columns = df.drop(non_tag_columns, axis=1)

# 统计所有标签列中每个非NaN值的出现次数，同时记录每个值对应的标签列
value_tag_counts = tag_columns.stack().reset_index()
value_tag_counts.columns = ['row', 'tag', 'value']
value_tag_counts = value_tag_counts.dropna(subset=['value'])
value_counts = value_tag_counts.groupby(['value', 'tag']).size().reset_index(name='count')

# 找出出现次数最多的前20个值
top_20_values = value_counts.groupby('value')['count'].sum().nlargest(50).reset_index()

# 只保留前20个值的数据
top_20_value_details = value_counts[value_counts['value'].isin(top_20_values['value'])]

# 输出结果到CSV文件
output_file = 'top_50_values_tags_counts.csv'
top_20_value_details.to_csv(output_file, index=False)

print(f"Top 50 values and their counts with tags written to {output_file}")
