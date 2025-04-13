import pandas as pd
import xml.etree.ElementTree as ET

# 读取用户上传的xlsx文件
xlsx_file = 'output_with_way_ids_processed (1).xlsx'
df = pd.read_excel(xlsx_file)

# 提取way ids，从第四列开始，确保转换为字符串并去掉.0
way_ids = df.iloc[1:, 3].dropna().astype(int).astype(str).unique().tolist()
print("提取的way IDs:", way_ids)

# 读取OSM文件路径
osm_file = 'Itami.osm'  # 更新为你的OSM文件路径
tree = ET.parse(osm_file)
root = tree.getroot()

# 构建way_id到way元素的映射
way_id_to_element = {way.attrib['id']: way for way in root.findall('way')}
print(f"Way elements mapped: {len(way_id_to_element)}")

# 定义函数，获取特定way id的所有标签
def get_tags(way_id):
    tags = {}
    way = way_id_to_element.get(way_id)
    if way is not None:
        for e in way:
            if e.tag == "tag":
                tags[e.attrib['k']] = e.attrib['v']
    return tags

# 找到所有除了给定way_ids之外的way及其标签
all_other_ways = []
for way_id, way in way_id_to_element.items():
    if way_id not in way_ids:
        tags = get_tags(way_id)
        way_data = {'way_id': way_id}
        way_data.update(tags)
        all_other_ways.append(way_data)

# 将其他ways的标签信息转换为DataFrame
other_ways_df = pd.DataFrame(all_other_ways)

# 检查是否正确填充了DataFrame
print(other_ways_df.head())

# 输出结果到新的xlsx文件，使用xlsxwriter加快速度
output_other_ways_file = 'output_other_ways_with_tags4444.xlsx'
with pd.ExcelWriter(output_other_ways_file, engine='xlsxwriter') as writer:
    other_ways_df.to_excel(writer, index=False)
print(f"Output written to {output_other_ways_file}")
