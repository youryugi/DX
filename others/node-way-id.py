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

# 输出结果
for node_id in node_ids:
    if node_id in node_to_way:
        print(f'Node ID {node_id} belongs to Way IDs: {", ".join(node_to_way[node_id])}')
    else:
        print(f'Node ID {node_id} not found in any Way IDs')
