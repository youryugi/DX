import osmium as osm
import pandas as pd
from scipy.spatial import cKDTree

class NodeHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.nodes = []
        self.node_tags = {}

    def node(self, n):
        self.nodes.append({
            'id': n.id,
            'latitude': n.location.lat,
            'longitude': n.location.lon
        })
        self.node_tags[n.id] = {tag.k: tag.v for tag in n.tags}

# 创建处理器实例
handler = NodeHandler()

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
input_df['tags'] = input_df['nearest_node_id'].apply(lambda node_id: handler.node_tags.get(node_id, 'No Tags'))

# 保存结果到 Excel 文件
input_df.to_excel('output_with_nearest_nodes_and_tags.xlsx', index=False)

print("已找到最近的节点并保存到 output_with_nearest_nodes_and_tags.xlsx 文件中")
