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

# 找到所有需要的标签键
all_tags = set()
for way_id in way_ids:
    tags = get_tags(way_id)
    print(f"Tags for way_id {way_id}: {tags}")  # 调试输出
    all_tags.update(tags.keys())

# 在原始DataFrame中添加新的列来存储way的属性
for tag in all_tags:
    df[tag] = ''

# 填充DataFrame
for i, row in df.iterrows():
    if i == 0:
        continue  # 跳过标题行
    if pd.isna(row[3]):
        print(f"Skipping row {i} due to NaN value")
        continue  # 跳过包含NaN的行
    way_id = str(int(row[3]))  # 确保way_id是字符串并去掉.0
    print(f"Current way_id: {way_id} (type: {type(way_id)})")  # 确认way_id的类型
    tags = get_tags(way_id)
    print(f"Processing way_id {way_id} with tags: {tags}")  # 添加调试输出
    for tag, value in tags.items():
        df.at[i, tag] = value

# 检查是否正确填充了DataFrame
print(df.head())

# 输出结果到新的xlsx文件
output_file = 'output_with_way_ids_and_tags_final.xlsx'
df.to_excel(output_file, index=False)
print(f"Output written to {output_file}")
