import pandas as pd
import xml.etree.ElementTree as ET

# 读取用户上传的xlsx文件
xlsx_file = 'output_other_ways_with_ids_without_acc.xlsx'
df = pd.read_excel(xlsx_file)

# 提取way ids，从第一列开始，确保转换为字符串
way_ids = df.iloc[:, 0].dropna().astype(int).astype(str).tolist()

# 读取OSM文件路径
osm_file = 'Itami.osm'  # 更新为你的OSM文件路径
tree = ET.parse(osm_file)
root = tree.getroot()

# 构建way_id到way元素的映射
way_id_to_element = {way.attrib['id']: way for way in root.findall('way')}


# 定义函数，获取特定way id的所有标签
def get_tags(way_id):
    tags = {}
    way = way_id_to_element.get(way_id)
    if way is not None:
        for e in way:
            if e.tag == "tag":
                tags[e.attrib['k']] = e.attrib['v']
    return tags


# 初始化结果列表
results = []

# 处理每个way id并获取标签
for i, way_id in enumerate(way_ids):
    tags = get_tags(way_id)
    way_data = {'way_id': way_id}
    way_data.update(tags)
    results.append(way_data)

    # 每写入100个输出一次进度
    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1} 个way ID")

# 将结果转换为DataFrame
result_df = pd.DataFrame(results)

# 输出结果到新的csv文件
output_file = 'output_with_tags.csv'
result_df.to_csv(output_file, index=False)

print(f"Output written to {output_file}")