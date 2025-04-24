import osmium as osm

class NodeHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.nodes = []

    def node(self, n):
        self.nodes.append({
            'id': n.id,
            'latitude': n.location.lat,
            'longitude': n.location.lon,
            'tags': {tag.k: tag.v for tag in n.tags}
        })

# 创建处理器实例
handler = NodeHandler()

# 读取 OSM 文件
handler.apply_file('Itami.osm')

# 输出节点信息
for node in handler.nodes:
    print(f"ID: {node['id']}, Latitude: {node['latitude']}, Longitude: {node['longitude']}, Tags: {node['tags']}")
