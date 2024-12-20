"""
This file contains a python scripts which does the following:
    - Initialises CSV files for the 64 regions
    - For each of the subregion we create a parallel thread, which is reponsible for the following actions:
        - Figuring out lat lon of 2km x 2km chunks for each row
        - For each lat which contains 682 such chunks, we will extract the features using resnet50
        - then using variance thresholding selected features we select 100 features and populate them in the csv files
"""

import time
import rasterio
import math
import numpy as np
from PIL import Image
import os
import pandas as pd
import torch
from torchvision import transforms
import threading
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import threading
import csv
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

torch.manual_seed(0)
np.random.seed(0)

def lat_long_to_pixel(lat, lon, img_width, img_height):
    """
    Converts latitude and longitude to pixel coordinates in the image.
    Assumes the image is georeferenced from (-180W, 90N) to (180E, -90S).
    """
    x = min(int((lon + 180) / 360 * img_width), img_width - 125)
    y = min(int(((90 - lat) / 180) * img_height), img_height - 125)
    return x, y

shapefile_path = './Mare Classfication/LROC_GLOBAL_MARE_180.SHP'
gdf = gpd.read_file(shapefile_path)
gdf['region_type'] = gdf['MARE_NAME'].apply(lambda x: 1 if pd.notnull(x) else 2)
tiff_image_path = "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
time1 = time.time()
with rasterio.open(tiff_image_path) as dataset:
    img_width = dataset.width
    img_height = dataset.height
    resolution_km = 0.1  # 100m/px corresponds to 0.1km/px
    image_array = dataset.read(1)
time2 = time.time()
print(f"[INFO] Loading of tiff img done in: {time2 - time1:.2f} seconds")    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_array = torch.tensor(image_array)

LATITUDE_RANGE = (90, -90)
LONGITUDE_RANGE = (-180, 180)
NUM_SUBREGIONS = 64
SUBREGIONS_PER_ROW = 8  # Assume the regions are divided into 8x8 grid
SQUARE_SIZE_KM = 2      # Each square's size in kilometers
MOON_RADIUS_KM = 1737.4 # Approximate radius of Moon in kilometers

def km_to_degrees(km):
    return km / (2 * np.pi * MOON_RADIUS_KM) * 360
LAT_PER_REGION = ((LATITUDE_RANGE[1] - LATITUDE_RANGE[0]) / SUBREGIONS_PER_ROW)
LON_PER_REGION = ((LONGITUDE_RANGE[1] - LONGITUDE_RANGE[0]) / SUBREGIONS_PER_ROW)
square_size_deg = km_to_degrees(SQUARE_SIZE_KM)
num_squares_lat = abs(int(LAT_PER_REGION // square_size_deg))
num_squares_lon = abs(int(LON_PER_REGION // square_size_deg))
# print(num_squares_lat, num_squares_lon)

elements = ['Fe', 'Ti', 'Ca', 'Si', 'Al', 'Mg', 'Na', 'O']
mareOrHighland = ['mareOrHighland']
topFeatures = [0, 1846, 1808, 1813, 1146, 1378, 923, 1237, 1558, 37, 1574, 1117, 103, 505, 550, 1734, 1785, 
               881, 1030, 1820, 1978, 792, 1323, 51, 1714, 691, 978, 1746, 1499, 1183, 1160, 1288, 371, 985, 
               34, 1696, 1101, 469, 1406, 133, 703, 1679, 258, 857, 1245, 914, 184, 157, 1988, 1641, 947, 
               1847, 1953, 2007, 787, 129, 793, 188, 163, 1262, 800, 1131, 1390, 66, 700, 590, 662, 916, 
               1538, 1673, 995, 1424, 139, 652, 959, 1869, 228, 1293, 1105, 1457, 2015, 692, 149, 1958, 
               647, 1530, 1228, 930, 567, 1003, 46, 1341, 1045, 1560, 741, 1995, 522, 1728, 1298, 783, 
               1778, 1077, 640, 1774, 226, 1694, 285, 969, 97, 1863, 578, 558, 780, 813, 397, 643, 696, 
               1026, 434, 559, 1699, 1195, 251, 534, 555, 1555, 1676, 403, 1373, 577, 762, 912, 1611, 
               943, 278, 1135, 1584, 1207, 323, 186, 1076, 1470, 1564, 952, 221, 1184, 419, 478, 880, 
               1276, 1938, 982, 1159, 116, 395, 1936, 1926, 980, 729, 524, 1290, 252, 1670, 264, 1727, 
               1083, 412, 398, 1155, 814, 688, 1865, 126, 561, 1835, 1372, 1154, 716, 362, 216, 1534, 
               320, 463, 866, 932, 843, 311, 672, 170, 1218, 869, 1665, 975, 1144, 110, 1946, 1691, 
               1698, 759, 761, 53, 1111, 1141, 1109, 457, 573, 2011, 1593, 25, 360, 650, 997, 1431, 
               347, 769, 427, 704, 1587, 1522, 262, 715, 746, 772, 1650, 354, 1458, 106, 840, 585, 
               353, 1110, 818, 1878, 422, 543, 637, 571, 1852, 826, 361, 1442, 1243, 1922, 656, 95, 
               1244, 1556, 1009, 1966, 1552, 456, 1463, 1363, 808, 1326, 481, 1468, 407, 2029, 1687, 
               974, 1898, 162, 917, 576, 1187, 1327, 1708, 1726, 57, 390, 1553, 1595, 710, 152, 91, 
               659, 1975, 273, 156, 701, 777, 1118, 1319, 105, 26, 1972, 1093, 1220, 833, 1776, 366, 
               306, 498, 1368, 31, 918, 1639, 1236, 1797] #top k features as discussed in top_k_features.ipynb 

wCount = len(topFeatures)
headers = ['lat_center', 'lon_center'] + elements + mareOrHighland + [f'w_{i}' for i in range(1, wCount + 1)]

target_size = (125, 125)
transform = transforms.Compose([
    transforms.Resize(target_size),  # Resize all images to (125, 125)
])
def extractFeaturesFromImages(images, resnet50):
    tensors = [transform(img) for img in images]
    # Stack tensors to create a single tensor with shape (682, 125, 125)
    tensor_stack = torch.stack(tensors).to(device)

    features_list = []
    for batch_start in range(0, len(tensor_stack), 64):
        batch = tensor_stack[batch_start:batch_start + 64]
        batch = batch.unsqueeze(1).repeat(1, 3, 1, 1).float().normal_()  # Repeat to match input channels
        batch_features = resnet50(Variable(batch)).data

        features_list.append(batch_features)
        torch.cuda.empty_cache()  # Clear cache after each batch
 
    features = torch.cat(features_list, dim=0)
    features = features.squeeze(-1).squeeze(-1).detach()
    features = features[:, topFeatures]

    return features

def isMareOrHighland(lat, lon):
    point = Point(lon, lat)
    possible_matches_index = list(gdf.sindex.intersection(point.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    is_maria = possible_matches['geometry'].apply(lambda x: x.contains(point)).any()
    
    return 1 if is_maria else 2

def classify_points(lat, lon_list):
    """
    Classifies a continuous set of points along the same latitude.
    lat: latitude of all points
    lon_list: list of longitudes
    """
    points = [Point(lon, lat) for lon in lon_list]
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)
    
    # Perform spatial join
    joined = gpd.sjoin(points_gdf, gdf[['geometry', 'region_type']], how='left', predicate='within')
    classifications = joined['region_type'].fillna(2).tolist()
    
    return classifications

def RegionProcessor(i, j):

    print(f"[INFO] Region {i} {j}: Starting")

    resnet50 = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    modules=list(resnet50.children())[:-1]
    resnet50 = nn.Sequential(*modules)
    for p in resnet50.parameters():
        p.requires_grad = False
    resnet50 = resnet50.to(device)

    subregion_lat = LATITUDE_RANGE[0] + i * LAT_PER_REGION
    subregion_lon = LONGITUDE_RANGE[0] + j * LON_PER_REGION
    filename = f"subregion_{i}_{j}.csv"
    subregion_ul_x, subregion_ul_y = lat_long_to_pixel(subregion_lat, subregion_lon, img_width, img_height)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        # Write rows for each square within the subregion
        for lat_idx in range(num_squares_lat):
            print(f"[INFO] Latitude {subregion_lat - lat_idx * square_size_deg} in subregion {i} {j}: Starting")
            images = []
            centersLAT = []
            centersLON = []
            mareOrHighlandList = []
            height_km = 12.5
            width_km = 12.5
            height_px = int(height_km / resolution_km)
            width_px = int(width_km / resolution_km)
            for lon_idx in range(num_squares_lon):
                ulx = min(max(subregion_ul_x + lon_idx*20 - (1 + 125//2), 0), img_width - 125)
                uly = min(max(subregion_ul_y + lat_idx*20 - (1 + 125//2) , 0), img_height - 125)

                sub_image_array = image_array[
                    uly : uly + height_px, ulx : ulx + width_px
                ]
                center_lat = subregion_lat - lat_idx*square_size_deg
                center_lon = subregion_lon + lon_idx*square_size_deg
                images.append(sub_image_array)
                centersLAT.append(center_lat)
                centersLON.append(center_lon)

            mareOrHighlandList = classify_points(subregion_lat - lat_idx * square_size_deg, centersLON)
            print(f"[INFO] Latitude {subregion_lat - lat_idx * square_size_deg} in subregion {i} {j}: Extracted {len(images)} Images")
            
            features = extractFeaturesFromImages(images, resnet50)
            
            print(f"[INFO] Latitude {subregion_lat - lat_idx * square_size_deg} in subregion {i} {j}: Calculated Features")

            for lon_idx in range(num_squares_lon):
                row = [centersLAT[lon_idx], centersLON[lon_idx]] + [0]*(len(elements)) + [mareOrHighlandList[lon_idx]] + features[lon_idx].tolist()
                writer.writerow(row)

            print(f"[INFO] Latitude {subregion_lat - lat_idx * square_size_deg} in subregion {i} {j}: Complete Write")
            del features
            del images
            del centersLAT
            del centersLON

    print(f"[INFO] Region {i} {j}: Completed")
    return

for i in range(SUBREGIONS_PER_ROW):
    threads = []
    for j in range(SUBREGIONS_PER_ROW):
        thread = threading.Thread(target=RegionProcessor, args=(i, j))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
print("[INFO] All subregion processing completed.")
