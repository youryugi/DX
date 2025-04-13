import pandas as pd
import xml.etree.ElementTree as ET

# 读取xlsx文件
xlsx_file = 'output_with_nearest_nodes.xlsx'
df = pd.read_excel(xlsx_file)

# 提取第三列的node id，从第二行开始提取
node_ids = set(df.iloc[1:, 2].astype(str).tolist())

# 构建node_id到way_id的索引
node_to_way = {}

# 逐行解析OSM文件
osm_file = 'Itami.osm'
context = ET.iterparse(osm_file, events=('start', 'end'))

# 初始化当前处理的way_id
current_way_id = None

for event, elem in context:
    if event == 'start':
        if elem.tag == 'way':
            current_way_id = elem.attrib['id']
    elif event == 'end':
        if elem.tag == 'way':
            # 清理已处理的元素
            elem.clear()
        elif elem.tag == 'nd':
            node_ref = elem.attrib['ref']
            if node_ref in node_ids:
                if node_ref not in node_to_way:
                    node_to_way[node_ref] = []
                node_to_way[node_ref].append(current_way_id)
            elem.clear()

# 创建新的DataFrame来存储结果
new_df = pd.DataFrame(columns=df.columns.tolist() + ['Way ID'])

# 填充新的DataFrame
for i, row in df.iterrows():
    if i == 0:
        new_df = pd.concat([new_df, pd.DataFrame([row], columns=new_df.columns)], ignore_index=True)
        continue
    node_id = str(row[2])
    if node_id in node_to_way:
        for way_id in node_to_way[node_id]:
            new_row = row.tolist() + [way_id]
            new_df = pd.concat([new_df, pd.DataFrame([new_row], columns=new_df.columns)], ignore_index=True)
    else:
        new_row = row.tolist() + ['']
        new_df = pd.concat([new_df, pd.DataFrame([new_row], columns=new_df.columns)], ignore_index=True)

# 输出结果到新的xlsx文件
output_file = 'output_with_way_ids111.xlsx'
new_df.to_excel(output_file, index=False)

print(f'Results have been written to {output_file}')
