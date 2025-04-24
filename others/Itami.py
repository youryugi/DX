import xml.etree.ElementTree as ET
from math import sqrt
import pandas as pd
import random
import time
import csv
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

# --------------Import Export------------------------#
global G_OSM_Node_List
global G_Itami_Accident
global G_OSM_Ways_List

class Node:
    id = 0
    lon = 0.0
    lat = 0.0
    def __init__(self, id, lat, lon):
        self.id = id
        self.lat = lat
        self.lon = lon

def load_Itami_Nodes(inputfile='Itami.osm',picklefile='Itami-Nodes.pckl'):
  global G_OSM_Node_List
  print("Loading: OSM_Node_List from",inputfile)
  OSM_Node_List=[]
  if os.path.isfile(picklefile) :
    with open(picklefile, 'rb') as f:
      OSM_Node_List = pickle.load(f)
  else:
    start = time.time()
    tree = ET.parse(inputfile)
    root = tree.getroot()
    for child in root:
      if child.tag == "node":
        lon = float(child.attrib['lon'])
        lat = float(child.attrib['lat'])
        #  minlon="135.3468000" maxlon="135.5503000" minlat="34.7258000" maxlat="34.8161000" /
        if 135.371005 < lon and lon < 135.4398981 and 34.75762528 < lat and lat < 34.81522806:
          OSM_Node_List.append(Node(int(child.attrib['id']),float(child.attrib['lon']),float(child.attrib['lat'])))
    with open(picklefile, 'ab') as f:
      pickle.dump(OSM_Node_List, f)
    end = time.time()
    print("time spent:",end - start,'\tNodes found:',len(OSM_Node_List))
  G_OSM_Node_List = OSM_Node_List
  return OSM_Node_List

def load_Accident(inputfile="Accidents.csv"):
  global G_Itami_Accident
  print("Loading: Itami_Accident from",inputfile)
  Itami_Accident = pd.read_csv(inputfile)
  G_Itami_Accident = Itami_Accident
  return Itami_Accident

class Way:
    id = 0
    nodes = []
    name = ""
    def __init__(self, id, nodes, name):
        self.id = id
        self.nodes = nodes
        self.name = name

def load_Itami_Ways(inputfile='Itami.osm',picklefile='Itami-Ways.pckl'):
  global G_OSM_Ways_List
  print("Loading: OSM_Ways_List from",inputfile)
  OSM_Ways_List = []
  if os.path.isfile(picklefile) :
    with open(picklefile, 'rb') as f:
      OSM_Ways_List = pickle.load(f)
  else:
    start = time.time()
    tree = ET.parse(inputfile)
    root = tree.getroot()
    for child in root:
      if child.tag == "way":
        ID=int(child.attrib['id'])
        nd=[]
        name=""
        for e in child:
          if e.tag == "nd":
            nd.append(int(e.attrib['ref']))
          else: #if e.tag == "tag":
            if e.attrib['k'] == "name" or e.attrib['k'] == "name:en":
              name = e.attrib['v']
              break
        OSM_Ways_List.append( Way(ID,nd,name) )
    with open(picklefile, 'ab') as f:
      pickle.dump(OSM_Ways_List, f)
    end = time.time()
    print("time spent:",end - start)
  G_OSM_Ways_List = OSM_Ways_List
  return OSM_Ways_List

# ---------------------------------------------------#

def distance(P1,P2: Node):
    x=P1['latitude']
    y=P1['longitude']
    a=P2.lat
    b=P2.lon
    return sqrt((b-y)**2 + (a-x)**2)

def get_closest_node(df_iter):
  row_index,accident_row=df_iter
  if row_index % 10 == 0:
    print(row_index,' ',end='',sep='')
  smallest_distance = 99999
  node_index=0
  for index,node_value in enumerate(G_OSM_Node_List):
    dist=distance(accident_row,node_value)
    if dist < smallest_distance:
      node_index=index
      smallest_distance=dist
  if smallest_distance != 99999:
    return (node_index,accident_row['serial number'])
  return None

def get_index_of_used_nodes(max_jobs=8,picklefile='Itami-Used-Nodes-Index.pckl'):
  node_index_list = []
  print("Loading: node_index_list from OSM_Node_List & Itami_Accident")
  if os.path.isfile(picklefile) :
    with open(picklefile, 'rb') as f:
      node_index_list = pickle.load(f)
  else:
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
      node_index_list = set(executor.map(get_closest_node, G_Itami_Accident.iterrows()))
    node_index_list.discard(None)
    node_index_list = list(node_index_list)
    with open(picklefile, 'ab') as f:
      pickle.dump(node_index_list, f)
    end = time.time()
    print("time spent:",end - start)
    #Le temps mesure est de 5262 secondes, soi environ 1h30
  
  return node_index_list


def check_boundaries():
  max_lon=0
  min_lon=1000
  max_lat=0
  min_lat=1000
  i=0
  for nd in OSM_Node_List:
    (_,a,b)=nd
    max_lon=max(a,max_lon)
    min_lon=min(a,min_lon)
    max_lat=max(b,max_lat)
    min_lat=min(b,min_lat)
    if(i%100==0):
        print("MAX:",max_lon,max_lat," MIN:",min_lon,min_lat)
    i+=1
  print(i,"nodes analyzed")


def main():
    print("Loading Data...")
    OSM_Node_List  = load_Itami_Nodes()
    OSM_Ways_List  = load_Itami_Ways()
    Itami_Accident = load_Accident()
    used_nodes_index = get_index_of_used_nodes()
    print("Data ready !")



if __name__ == "__main__":
    main()
