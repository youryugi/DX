import osmium as osm
import pandas as pd
from scipy.spatial import cKDTree

class OSMHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.nodes = []
        self.node_tags = {}
        self.way_tags = {}

    def node(self, n):
        self.nodes.append({
            'id': n.id,
            'latitude': n.location.lat,
            'longitude': n.location.lon
        })
        self.node_tags[n.id] = {tag.k: tag.v for tag in n.tags}

    def way(self, w):
        tags = {tag.k: tag.v for tag in w.tags}
        for n in w.nodes:
            if n.ref not in self.node_tags:  # Only add tags if the node doesn't have its own
                self.node_tags[n.ref] = tags

# 创建处理器实例
handler = OSMHandler()

# 读取 OSM 文件
handler.apply_file('Itami.osm')

# 创建 pandas DataFrame 存储节点信息
nodes_df = pd.DataFrame(handler.nodes)

# 构建 KD 树用于快速最近邻搜索
kd_tree = cKDTree(nodes_df[['latitude', 'longitude']])

# 读取没有列头的输入表格并手动指定列名
input_df = pd.read_excel('itami2.xlsx', header=None)
input_df.columns = ['latitude', 'longitude']

# 找到每个输入点最近的节点
distances, indices = kd_tree.query(input_df[['latitude', 'longitude']])

# 获取最近节点的 ID
input_df['nearest_node_id'] = nodes_df.iloc[indices]['id'].values

# 获取最近节点的标签信息
def get_tags(node_id):
    if node_id in handler.node_tags:
        return handler.node_tags[node_id]
    return 'No Tags'

def get_way_tags(node_id):
    if node_id in handler.node_tags and 'No Tags' in handler.node_tags[node_id]:
        return handler.node_tags[node_id]['No Tags']
    return 'No Way Tags'

input_df['tags'] = input_df['nearest_node_id'].apply(lambda node_id: get_tags(node_id))
input_df['way_tags'] = input_df['nearest_node_id'].apply(lambda node_id: get_way_tags(node_id))

# 保存结果到 Excel 文件
input_df.to_excel('output_with_nearest_nodes_and_tags.xlsx', index=False)

print("已找到最近的节点并保存到 output_with_nearest_nodes_and_tags.xlsx 文件中")
