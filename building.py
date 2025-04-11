import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import xml.etree.ElementTree as ET


def parse_gml(file_path):
    """
    Parse the GML file and extract all 3D polygons representing buildings.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {
        'gml': 'http://www.opengis.net/citygml/bridge/2.0',
        # Adjust namespaces as needed
    }

    buildings = []

    for polygon in root.findall('.//gml:Polygon', ns):
        # Extract all positions within the polygon
        coords = []
        for pos in polygon.findall('.//gml:pos', ns):
            coord = list(map(float, pos.text.split()))
            coords.append(coord)
        if coords:
            buildings.append(coords)

    return buildings


def plot_buildings(buildings):
    """
    Plot all buildings in a 3D plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for building in buildings:
        # Convert coordinates to 3D polygons
        x, y, z = zip(*building)
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts, alpha=0.5, edgecolor='k')
        ax.add_collection3d(poly)

    # Set axes labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title("3D Visualization of Buildings")
    plt.show()


# File path to your GML file
file_path = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\52350386_bldg_6697_op.gml"

# Parse GML file
buildings = parse_gml(file_path)

# Plot all buildings
if buildings:
    plot_buildings(buildings)
else:
    print("No buildings found in the GML file.")
