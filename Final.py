 
# # Region Superresolution Code
# 
# This python notebook runs the complete process for a dynamic updation starting from abundance population, and graph finetuning.
# 
# Code Flow:
# - Input Parameters: input_file.csv
# - Output: subregion_i_j.csv containing all 2km x 2km elemental abundances for that region.

from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import re
import pandas as pd
from torchvision import transforms
import geopandas as gpd
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch_optimizer as optim
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch.autograd import Variable
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

def current_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

torch.manual_seed(0)
np.random.seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(f"[INFO] {current_time()} Imports done. Using {device}")
os.makedirs('regions', exist_ok=True)


# # All mare files
# !gdown 1Ig5nXZqwscWYRLWgD22a76N5AsIDF2cv
# !gdown 1RgGrRBntdAb6aMf7mvAgD23WunY2uqv8
# !gdown 1LqYxaY08nyZqGlK0Udd28MzekwzwLsCz
# # !gdown 1794ulttX9D46Onumwvg6ToBJCGd9M05L
# !gdown 15DUqZp716VV60jBM8V0CCg2oQdlBXlDU
# !gdown 1Ltm2PjQvXCQ-NiJjARWrJGsr74AxWv3g
# !gdown 1agqOartoVDRJ65yG_gxmtF6Z6oyk7Xd4
# !gdown 1ul-A8zHvUUOl62F0fzDqm8wL49GNuDiL


# # All region files
# !gdown --folder --output ./regions/ 1O9Lhg54pWAxzRc4OuITKIKOVlftkHEFV
# !gdown --folder --output ./regions/ 1Vea75zfr8pns5SnMf3Uxseqsd822KqnU
# !gdown --folder --output ./regions/ 1G4hAxJ_cvsEmBWBreA2b8t69L42NyFIm
# !gdown --folder --output ./regions/ 1shlmJDzid368w8XvAW8Q1in49SHhenDH

# # Part 1: Some functions and constants
MOON_RADIUS_KM = 1737.4 # Approximate radius of Moon in kilometers
# Define constants
LATITUDE_RANGE = (90, -90)
LONGITUDE_RANGE = (-180, 180)
NUM_SUBREGIONS = 64
SUBREGIONS_PER_ROW = 8  # Assume the regions are divided into 8x8 grid
SQUARE_SIZE_KM = 2      # Each square's size in kilometers
# Calculate the size of each subregion in degrees in terms of equator
LAT_PER_REGION = ((LATITUDE_RANGE[1] - LATITUDE_RANGE[0]) / SUBREGIONS_PER_ROW)
LON_PER_REGION = ((LONGITUDE_RANGE[1] - LONGITUDE_RANGE[0]) / SUBREGIONS_PER_ROW)
def km_to_degrees(km):
    return km / (2 * np.pi * MOON_RADIUS_KM) * 360
square_size_deg = km_to_degrees(SQUARE_SIZE_KM)
num_squares_lat = abs(int(LAT_PER_REGION // square_size_deg))
num_squares_lon = abs(int(LON_PER_REGION // square_size_deg))
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
               306, 498, 1368, 31, 918, 1639, 1236, 1797]
wCount = len(topFeatures)
headers = ['lat_center', 'lon_center'] + elements + mareOrHighland + [f'w_{i}' for i in range(1, wCount + 1)]

# Lat Long to Pixel --> Converts latitude longitude to pixel coordinate in the image array
def lat_long_to_pixel(lat, lon, img_width, img_height):
    """
    Converts latitude and longitude to pixel coordinates in the image.
    Assumes the image is georeferenced from (-180W, 90N) to (180E, -90S).
    """
    x = min(int((lon + 180) / 360 * img_width), img_width - 125)
    y = min(int(((90 - lat) / 180) * img_height), img_height - 125)
    return x, y
# Load the Mare Areas Shapefile
shapefile_path = './LROC_GLOBAL_MARE_180.SHP'
gdf = gpd.read_file(shapefile_path)
gdf['region_type'] = gdf['MARE_NAME'].apply(lambda x: 1 if pd.notnull(x) else 2)
gdf.sindex  # Creates a spatial index if not already present
print(f"[INFO] {current_time()} Done importing geodataframe for highland-mare classification")

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
    classifications = joined['region_type'].fillna(2).tolist()  # Default to 2 (Highland)

    return classifications

def isMareOrHighland(lat, lon):
    '''
        Returns if the given coordinate belongs to a mare or highland region
        1 - Mare
        2 - Highland
    '''
    # Create a GeoDataFrame for the point of interest
    point = Point(lon, lat)  # Make sure lon, lat are in the correct order (lon, lat)
    gdf_point = gpd.GeoDataFrame(geometry=[point], crs=gdf.crs)

    # Check if the point lies within any of the polygons (maria regions)
    is_maria = gdf['geometry'].apply(lambda x: x.contains(point)).any()

    if is_maria:
        return 1
    else:
        return 2

 
# # Part 2: File Population

# This part exposes a function Part2(dataframe) responsible for reading the supplied abundances dataframe, solving the optimisation condition to get the abundances of the 8 subpoints in each rectangle. This is then populated in the subregion file. 
# It returns region wise number of updated points, number of updated points in each subregion of the region and the indices where each csv file is updated

lat_per_region = (LATITUDE_RANGE[0] - LATITUDE_RANGE[1]) / SUBREGIONS_PER_ROW
lon_per_region = (LONGITUDE_RANGE[1] - LONGITUDE_RANGE[0]) / SUBREGIONS_PER_ROW
regions = 0
updatedRegions = np.zeros((SUBREGIONS_PER_ROW, SUBREGIONS_PER_ROW))


def find_region_indices(lat, lon):
    """
    Determine which of the 64 regions a given latitude and longitude belongs to.

    Parameters:
    - lat: Latitude coordinate (90 to -90)
    - lon: Longitude coordinate (-180 to 180)

    Returns:
    - Tuple of (row_index, column_index) for the region (0-7, 0-7)
    """
    # Calculate row and column indices
    row_index = min(7, abs(int((LATITUDE_RANGE[0] - lat) / lat_per_region)))
    col_index = min(7, abs(int((lon - LONGITUDE_RANGE[0]) / lon_per_region)))

    return row_index, col_index

def find_subregion_indices(lat, lon):
    """
    Determines all of the 78 overlapping subregions of a region a given latitude and longitude belongs to.

    Parameters:
    - lat: Latitude coordinate (90 to -90)
    - lon: Longitude coordinate (-180 to 180)

    Returns:
    - list of tuples [(r_i, c_i)] where r_i belongs to (0 - 5) and c_i belongs to (0 - 12)
    """

    # Constants for the grid and chunk divisions
    LAT_GRID_SIZE = lat_per_region
    LON_GRID_SIZE = lon_per_region
    LAT_CHUNK_SIZE = LAT_GRID_SIZE / 342  # Degrees per chunk in latitude
    LON_CHUNK_SIZE = LON_GRID_SIZE / 682  # Degrees per chunk in longitude

    # Subregion starting points for latitude and longitude
    LAT_SUBREGION_STARTS = [0, 50, 100, 150, 200, 242]
    LON_SUBREGION_STARTS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 582]
    LAT_SUBREGION_ENDS = [start + 100 for start in LAT_SUBREGION_STARTS]
    LON_SUBREGION_ENDS = [start + 100 for start in LON_SUBREGION_STARTS]

    # Step 1: Identify the grid cell
    grid_row = min(7, abs(int((LATITUDE_RANGE[0] - lat) / lat_per_region)))
    grid_col = min(7, abs(int((lon - LONGITUDE_RANGE[0]) / lon_per_region)))

    # Step 2: Relative position within the grid cell
    rel_lat = (90 - lat) % lat_per_region
    rel_lon = (lon + 180) % lon_per_region

    # Step 3: Map to chunk coordinates within the 342x682 grid
    chunk_row = int(rel_lat // LAT_CHUNK_SIZE)
    chunk_col = int(rel_lon // LON_CHUNK_SIZE)

    # Step 4: Determine overlapping subregions
    subregion_indices = []
    for r_idx, (start_lat, end_lat) in enumerate(zip(LAT_SUBREGION_STARTS, LAT_SUBREGION_ENDS)):
        if start_lat <= chunk_row < end_lat:
            for c_idx, (start_lon, end_lon) in enumerate(zip(LON_SUBREGION_STARTS, LON_SUBREGION_ENDS)):
                if start_lon <= chunk_col < end_lon:
                    subregion_indices.append((r_idx, c_idx))

    return grid_row, grid_col, subregion_indices

def find_and_update_subregion_indices(lat, lon, updatedRegions, updatedSubregions):
    row_index, col_index, subregion_indices = find_subregion_indices(lat, lon)

    updatedRegions[row_index][col_index] += 1
    for tup in subregion_indices:
        updatedSubregions[row_index][col_index][tup[0]][tup[1]] += 1

    return row_index, col_index, updatedRegions, updatedSubregions


def calculate_abundances(A, labels, counter):
    """
    Calculate abundances using the optimization method

    Parameters:
    - A: Input tensor of initial abundances (8x1)
    - labels: Labels for 8 regions (1 for Mare, otherwise Highland)

    Returns:
    - Calculated abundances matrix (C)
    """
    # print(f"[DEBUG] Calculating abundances for label : {labels} : done : {counter}")
    # Highland and Mare reference abundances
    # ["Fe", "Ti", "Ca", "Si", "Al", "Mg", "Na", "O"]
    Ah = torch.tensor([2.58, 0.66, 13.15, 22.41, 13.65, 0.18, 1.37, 45], dtype=torch.float32).reshape(8, 1)
    Am = torch.tensor([9.72, 4.37, 7.00, 18.76, 7.40, 7.23, 1.37, 45], dtype=torch.float32).reshape(8, 1)

    # Matrix D (8x8): each column is Ah or Am based on labels
    D = torch.zeros(8, 8, dtype=torch.float32)
    for i, label in enumerate(labels):
        D[:, i] = Ah.squeeze() if label == 2 else Am.squeeze()

    # Regularization parameter
    lambda_reg = 1.0
    lambda_nonneg = 2.0  # Penalty for negative values

    # Define Lagrangian Function
    def lagrangian(B_flat):
        B = B_flat.reshape(1, 8)  # Reshape B into 1x8
        C = A @ B                # Compute C = A * B (8x8)
        loss = torch.norm(C - D, p='fro') ** 2
        constraint = (torch.mean(C, dim=1) - A.squeeze()) ** 2
        non_negativity_penalty = torch.sum(torch.relu(-C))
        lagrangian_value = loss + lambda_reg * torch.sum(constraint) + lambda_nonneg * non_negativity_penalty
        return lagrangian_value

    B = torch.rand(1, 8, requires_grad=True)
    optimizer = torch.optim.Adam([B], lr=0.01)

    num_epochs = 80
    loss_history = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = lagrangian(B)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    C = torch.relu(A @ B.detach())
    return C


def interpolate_subregion_center(vertices_lat, vertices_lon, i, divide_along_left_right):
    """
    Interpolate the center coordinates for a specific subregion along the longer side.

    Parameters:
    - vertices_lat: List of 4 vertex latitudes in clockwise order.
    - vertices_lon: List of 4 vertex longitudes in clockwise order.
    - i: Subregion index (0-7).
    - divide_along_left_right: Boolean, True if dividing along left-right sides, False if top-bottom.

    Returns:
    - Tuple of (subregion_center_lat, subregion_center_lon).
    """
    def wrap_longitude(lon):
        """Wrap longitude to [-180, 180]."""
        return (lon + 180) % 360 - 180

    if divide_along_left_right:
        left_lat_interp = vertices_lat[0] + (i + 0.5) * (vertices_lat[3] - vertices_lat[0]) / 8
        left_lon_interp = vertices_lon[0] + (i + 0.5) * (wrap_longitude(vertices_lon[3] - vertices_lon[0])) / 8

        right_lat_interp = vertices_lat[1] + (i + 0.5) * (vertices_lat[2] - vertices_lat[1]) / 8
        right_lon_interp = vertices_lon[1] + (i + 0.5) * (wrap_longitude(vertices_lon[2] - vertices_lon[1])) / 8

        subregion_center_lat = (left_lat_interp + right_lat_interp) / 2
        subregion_center_lon = wrap_longitude((left_lon_interp + right_lon_interp) / 2)

    else:  
        top_lat_interp = vertices_lat[0] + (i + 0.5) * (vertices_lat[1] - vertices_lat[0]) / 8
        top_lon_interp = vertices_lon[0] + (i + 0.5) * (wrap_longitude(vertices_lon[1] - vertices_lon[0])) / 8

        bottom_lat_interp = vertices_lat[3] + (i + 0.5) * (vertices_lat[2] - vertices_lat[3]) / 8
        bottom_lon_interp = vertices_lon[3] + (i + 0.5) * (wrap_longitude(vertices_lon[2] - vertices_lon[3])) / 8

        subregion_center_lat = (top_lat_interp + bottom_lat_interp) / 2
        subregion_center_lon = wrap_longitude((top_lon_interp + bottom_lon_interp) / 2)

    return subregion_center_lat, subregion_center_lon


def group_subregions(vertices_lat, vertices_lon):
    """
    Groups subregions based on the longer side of the quadrilateral.

    Parameters:
    - vertices_lat: List of latitudes of the region's vertices.
    - vertices_lon: List of longitudes of the region's vertices.

    Returns:
    - A dictionary with subregion indices as keys and center coordinates as values.
    """
    def great_circle_distance(lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        delta_lon = np.abs(lon2 - lon1)
        delta_lon = np.minimum(delta_lon, 2 * np.pi - delta_lon)
        central_angle = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon))
        return central_angle * MOON_RADIUS_KM  # Moon's radius in kilometers

    left_length = great_circle_distance(vertices_lat[0], vertices_lon[0], vertices_lat[3], vertices_lon[3])
    right_length = great_circle_distance(vertices_lat[1], vertices_lon[1], vertices_lat[2], vertices_lon[2])
    top_length = great_circle_distance(vertices_lat[0], vertices_lon[0], vertices_lat[1], vertices_lon[1])
    bottom_length = great_circle_distance(vertices_lat[3], vertices_lon[3], vertices_lat[2], vertices_lon[2])

    divide_along_left_right = (left_length + right_length) >= (top_length + bottom_length)
    subregion_centers = {}
    for i in range(8):
        subregion_center_lat, subregion_center_lon = interpolate_subregion_center(vertices_lat, vertices_lon, i, divide_along_left_right)
        subregion_centers[i] = (subregion_center_lat, subregion_center_lon)

    return subregion_centers

def find_grid_indices_in_subregion(lat, lon, subregion_row, subregion_col):
    """
    Calculate the grid indices for a given latitude and longitude within a subregion.

    Parameters:
    - lat: Latitude of the point.
    - lon: Longitude of the point.
    - subregion_row: Row index of the subregion.
    - subregion_col: Column index of the subregion.

    Returns:
    - Tuple (square_row, square_col) representing the grid indices.
    """

    subregion_lat_start = LATITUDE_RANGE[0] - subregion_row * lat_per_region
    subregion_lon_start = LONGITUDE_RANGE[0] + subregion_col * lon_per_region
    square_row = abs(int((lat - subregion_lat_start) // square_size_deg))
    square_col = abs(int((lon - subregion_lon_start) // square_size_deg))

    if not (0 <= square_row < num_squares_lat) or not (0 <= square_col < num_squares_lon):
        square_row = max(0, min(square_row, num_squares_lat))
        square_col = max(0, min(square_col, num_squares_lon))

    row_index = square_row * num_squares_lon + square_col

    return row_index

def save_subregion(subregion_row, subregion_col, points):
    print(f"Processing subregion: ({subregion_row}, {subregion_col})")

    local_indices = []

    csv_filename = f"./regions/ISRO_RegionData{subregion_row - subregion_row%2}{1+subregion_row - subregion_row%2}/subregion_{subregion_row}_{subregion_col}.csv"
    print(f"[INFO] {current_time()} Processing file: {csv_filename}")

    df = pd.read_csv(csv_filename)
    dtype_spec = {
        'lat_center': 'float',
        'lon_center': 'float',
        'Fe': 'float',
        'Ti': 'float',
        'Ca': 'float',
        'Si': 'float',
        'Al': 'float',
        'Mg': 'float',
        'Na': 'float',
        'O': 'float',
    }
    df = df.astype(dtype_spec)
    elements = ['Fe', 'Ti', 'Ca', 'Si', 'Al', 'Mg', 'Na', 'O']

    for point in points:
        # Find the closest grid index
        grid_index = find_grid_indices_in_subregion(
            point['x_center'], point['y_center'], subregion_row, subregion_col
        )
        if grid_index not in df.index:
            continue
        existing_values = df.loc[grid_index, elements]

        local_indices.append(grid_index)

        if existing_values.sum() == 0:
            df.loc[grid_index, elements] = [
                point['Fe'], point['Ti'], point['Ca'],
                point['Si'], point['Al'], point['Mg'], point['Na'], point['O']]
            #print(f"[INFO] {current_time()} Updated values at index {grid_index} in file {csv_filename} at {point['x_center'], point['y_center']}")
        else:
            new_values = pd.Series({
                'Fe': point['Fe'],
                'Ti': point['Ti'],
                'Ca': point['Ca'],
                'Si': point['Si'],
                'Al': point['Al'],
                'Mg': point['Mg'],
                'Na': point['Na'],
                'O': point['O'],
            })
            # df.loc[grid_index, elements] = (
            #     (existing_values + new_values) / 2
            # )
            df.loc[grid_index, elements] = new_values
            #print(f"[INFO] {current_time()} Averaged values at index {grid_index} in file {csv_filename} at {point['x_center'], point['y_center']}")

    df.to_csv(csv_filename, index=False)
    print(f"[INFO] {current_time()} Saved updates to file: {csv_filename}")

    return local_indices

def update_csv_files(subregion_data, indices):
    """
    Update existing CSV files for all subregions by populating the closest node
    to the subregion's center using grid-based calculation. Updates are based on:
    - Direct update if existing values are zero.
    - Averaging if existing values are non-zero.

    Parameters:
    - subregion_data: Dictionary with subregion keys and collected points as values.

    Returns:
    - indices: An 8x8 array of lists containing row_indices of each updation
    """

    # # for (subregion_row, subregion_col), points in tqdm(subregion_data.items(), desc="Processing Subregions"):
    # with ThreadPoolExecutor(max_workers=6) as executor:
    #     futures = [
    #         executor.submit(process_subregion, subregion_row, subregion_col, points)
    #         for (subregion_row, subregion_col), points in subregion_data.items()
    #     ]

    #     for future in tqdm(futures, desc="Processing Subregions"):
    #         future.result()  # Handle exceptions if any.

    subregion_tasks = list(subregion_data.items())  
    with ThreadPoolExecutor(max_workers = 2) as executor:
        # Initialize progress bar
        with tqdm(total=len(subregion_tasks), desc="Updating CSV Files") as pbar:
            futures = {
                executor.submit(save_subregion, subregion_row, subregion_col, points):
                (subregion_row, subregion_col)
                for (subregion_row, subregion_col), points in subregion_tasks
            }
            for future in as_completed(futures):
                subregion_row, subregion_col = futures[future]
                try:
                    local_indices = future.result()
                    indices[subregion_row][subregion_col].extend(local_indices)
                except Exception as e:
                    print(f"[ERROR] Subregion ({subregion_row}, {subregion_col}): {e}")
                pbar.update(1)

    return indices


def process_data_regions(data_regions, batch_size, updatedRegions, updatedSubregions, indices):
    """
    Process data regions with optimized subregion segregation and abundance calculation.

    Parameters:
    - data_regions: Original dataset containing 4-vertex coordinates and element abundances
    - batch_size: Number of data regions to process in one batch.
    - updatedRegions: a numpy array of shape 8, 8 to store number of enteries added in each region
    - updatedSubregions: a numpy array of shape 8, 8, 6, 13 to store number of enteries added in each subregion
    - indices: a numpy array of shape shape 8, 8 containing lists to store indices of updation

    This function collects data for subregions and updates CSVs after batch processing.
    """

    bools = input('Is it old data?')

    for batch_start in range(0, len(data_regions), batch_size):
        batch_regions = data_regions[batch_start:batch_start + batch_size]
        subregion_data = defaultdict(list)

        for counter, region in tqdm(batch_regions.iterrows(), total=len(batch_regions), desc="Calculating abundances"):
            # Extract vertex coordinates
            vertices_lat = [
                region['V0_lat'], region['V1_lat'],
                region['V2_lat'], region['V3_lat']
            ]
            vertices_lon = [
                region['V0_lon'], region['V1_lon'],
                region['V2_lon'], region['V3_lon']
            ]

            subregion_centers = group_subregions(vertices_lat, vertices_lon)
            initial_abundances = torch.tensor([
                region['Fe'], region['Ti'], region['Ca'],
                region['Si'], region['Al'], region['Mg'], region['Na'], region['O']
            ], dtype=torch.float32).reshape(8, 1)

            labels = [
                isMareOrHighland(subregion_center_lat, subregion_center_lon)
                for subregion_center_lat, subregion_center_lon in subregion_centers.values()
            ]

            if bools == 'yes':
                TI = np.random.normal(np.mean([4.37 if label == 1 else 0.66 for label in labels]), 1)
                region['Ti'] = TI
                initial_abundances[1] = TI
                CA = 100 - initial_abundances.sum().item()
                region['Ca'] = CA
                initial_abundances[2] = CA
            optimized_abundances = calculate_abundances(initial_abundances, labels, counter)

            for i, (subregion_center_lat, subregion_center_lon) in subregion_centers.items():
                subregion_row, subregion_col, updatedRegions, updatedSubregions = find_and_update_subregion_indices(subregion_center_lat, subregion_center_lon, updatedRegions, updatedSubregions)
                subregion_key = (subregion_row, subregion_col)

                subregion_data[subregion_key].append({
                    'x_center': subregion_center_lat,
                    'y_center': subregion_center_lon,
                    'Fe': optimized_abundances[0, i].item(),
                    'Ti': optimized_abundances[1, i].item(),
                    'Ca': optimized_abundances[2, i].item(),
                    'Si': optimized_abundances[3, i].item(),
                    'Al': optimized_abundances[4, i].item(),
                    'Mg': optimized_abundances[5, i].item(),
                    'Na': optimized_abundances[6, i].item(),
                    'O': optimized_abundances[7, i].item(),
                })

        for subregion_key, entries in subregion_data.items():
            print(f"Subregion '{subregion_key}' has {len(entries)} entries in batch {batch_start//batch_size}")
        indices = update_csv_files(subregion_data, indices)

        return updatedRegions, updatedSubregions, indices

def preprocess(data):
    # Clean column names by stripping leading/trailing whitespace
    data.columns = data.columns.str.strip()
    # Step 1: Extract only the weight columns for elements
    weight_columns = [col for col in data.columns if col.endswith('_WT')]
    lats = ['V0_LATITUDE', 'V0_LONGITUDE', 'V1_LATITUDE', 'V1_LONGITUDE', 'V2_LATITUDE', 'V2_LONGITUDE', 'V3_LATITUDE', 'V3_LONGITUDE']
    weights_data = data[weight_columns + lats].copy()
    # Step 2: Add fixed columns for "Na" and "O"
    weights_data['Na'] = 1.375
    weights_data['O'] = 45
    weights_data['Ti'] = 0
    weights_data['Ca'] = 0
    # Resulting dataset
    weights_data.head()
    weights_data.rename(columns={
        'V0_LATITUDE': 'lat0', 'V0_LONGITUDE': 'lon0',
        'V1_LATITUDE': 'lat1', 'V1_LONGITUDE': 'lon1',
        'V2_LATITUDE': 'lat2', 'V2_LONGITUDE': 'lon2',
        'V3_LATITUDE': 'lat3', 'V3_LONGITUDE': 'lon3',
        'MG_WT': 'Mg', 'AL_WT': 'Al', 'SI_WT': 'Si',
        'FE_WT': 'Fe'
    }, inplace=True)
    # Assuming `df` is your dataframe
    new_column_order = [
        'lat0', 'lon0', 'lat1', 'lon1', 'lat2', 'lon2', 'lat3', 'lon3',
        'Fe', 'Ti', 'Ca', 'Si', 'Al', 'Mg', 'Na', 'O'
    ]
    weights_data = weights_data[new_column_order]
    print(weights_data.head())
    # exit()
    return weights_data

def RegionProcessor2(file_path, batch_size, updatedRegions, updatedSubregions, indices):
    """
    Function to process a single data region file.
    """

    print(f"[INFO] {current_time()} Processing file: {file_path}")

    # Read the data region file
    data_regions = pd.read_csv(file_path)

    # data_regions = preprocess(data_regions)
    data_regions.rename(columns={
        'lat0': 'V0_lat', 'lon0': 'V0_lon',
        'lat1': 'V1_lat', 'lon1': 'V1_lon',
        'lat2': 'V2_lat', 'lon2': 'V2_lon',
        'lat3': 'V3_lat', 'lon3': 'V3_lon',
    }, inplace=True)

    # Process the data in batches
    updatedRegions, updatedSubregions, indices = process_data_regions(data_regions, batch_size, updatedRegions, updatedSubregions, indices)
    return updatedRegions, updatedSubregions, indices


# List of data region files
def Part2(file_name):
    batch_size = 3000

    updatedRegions = np.zeros((8, 8))
    updatedSubregions = np.zeros((8, 8, 6, 13))
    # Create an 8x8 array with empty lists
    indices = np.empty((8, 8), dtype=object)
    # Initialize each element to be an empty list
    for i in range(8):
        for j in range(8):
            indices[i, j] = []

    return RegionProcessor2(file_name, batch_size, updatedRegions, updatedSubregions, indices)

 
# Now for each input file, I have saved all enteries in the correct subregion file. I just need to process each input file now.

 
# # Part 3: CSV to Subgraphs
# This exposes a function (Part3), which given any i, j, iteration_number will create the subgraphs of all the 78 subsubregions and save them as a .pt file in the folder: ./graphs/subregion_i_j/subregion_i_j_x_y.pt, and also masks for each subsubregion listing the rows where original abundances are known.
# 
# This can be used in an iteration loop in Part 5 to train all graphs for a subregion (i, j)


# These parameters control graph construction and sliding window behavior
ALPHA = 1                 # Spatial distance weight factor
BETA = 1.2                # Feature distance weight factor
GAMMA = 1
K = 100                   # Maximum number of nearest neighbors to connect for each node
BATCH_SIZE = 1000         # Number of nodes processed in parallel batches
LAT_STEP = 682            # Number of longitude entries per latitude block
WINDOW_SIZE = 100         # Size of sliding window
STRIDE = 50               # Step size between sliding windows
LAT_SIZE = 342            # Total latitude grid size
LON_SIZE = 682            # Total longitude grid size
OccupancyMatrix = np.zeros((6, 13))


def haversine_gpu(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distances between geographical points using Haversine formula on GPU.

    Args:
        lat1 (torch.Tensor): Latitude of first point(s)
        lon1 (torch.Tensor): Longitude of first point(s)
        lat2 (torch.Tensor): Latitude of second point(s)
        lon2 (torch.Tensor): Longitude of second point(s)

    Returns:
        torch.Tensor: Distances between points in kilometers
    """
    R = MOON_RADIUS_KM  # mooon radius in kilometers

    # Convert degrees to radians
    lat1, lat2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    lon1, lon2 = torch.deg2rad(lon1), torch.deg2rad(lon2)

    # Haversine formula computation
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    return 2 * R * torch.arcsin(torch.sqrt(a))

def compute_spatial_distances_gpu(batch_lat, batch_lon, all_lat, all_lon):
    """
    Compute and normalize spatial distance matrix between a batch of points and all points.

    Args:
        batch_lat (torch.Tensor): Latitudes of the current batch
        batch_lon (torch.Tensor): Longitudes of the current batch
        all_lat (torch.Tensor): Latitudes of all points
        all_lon (torch.Tensor): Longitudes of all points

    Returns:
        torch.Tensor: Normalized spatial distance matrix
    """
    distances = haversine_gpu(batch_lat.unsqueeze(1), batch_lon.unsqueeze(1), all_lat, all_lon)
    min_val = distances.min()
    max_val = distances.max()
    normalized_distances = (distances - min_val) / (max_val - min_val)

    return normalized_distances

def compute_mare_highland_distances_gpu(batch_mare, all_mare):
    """
    Compute mare/highland distance matrix based on categorical difference.

    Args:
        batch_mare (torch.Tensor): Mare/Highland values (1 or 2) of the current batch
        all_mare (torch.Tensor): Mare/Highland values (1 or 2) of all nodes

    Returns:
        torch.Tensor: Binary mare/highland distance matrix (1 if different, 0 if same)
    """
    # Compare values to compute binary distance: 1 if different, 0 if same
    distances = (batch_mare.unsqueeze(1) != all_mare).float()
    return distances

def compute_feature_distances_gpu(batch_w, all_w):
    """
    Compute and normalize Euclidean feature distances between a batch of nodes and all nodes.

    Args:
        batch_w (torch.Tensor): Feature vectors of the current batch
        all_w (torch.Tensor): Feature vectors of all nodes

    Returns:
        torch.Tensor: Normalized feature distance matrix
    """
    distances = torch.cdist(batch_w, all_w, p=2)
    min_val = distances.min()
    max_val = distances.max()
    normalized_distances = (distances - min_val) / (max_val - min_val)

    return normalized_distances

def compute_edge_weights_gpu(Dspatial, Dfeature, Dmare, alpha, beta, gamma):
    """
    Compute edge weights based on spatial and feature distances.

    Uses an exponential decay function: exp(-Î± * spatial_dist - Î² * feature_dist)

    Args:
        Dspatial (torch.Tensor): Spatial distance matrix
        Dfeature (torch.Tensor): Feature distance matrix
        alpha (float): Spatial distance weight factor
        beta (float): Feature distance weight factor

    Returns:
        torch.Tensor: Edge weight matrix
    """
    return torch.exp(-alpha * Dspatial - beta * Dfeature - gamma * Dmare)

def calculate_sliding_windows(window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Calculate sliding window positions for a grid of specific dimensions.

    Args:
        window_size (int): Size of each sliding window
        stride (int): Step size between windows

    Returns:
        tuple: Latitude and longitude positions and sizes
    """
    def get_windows(num_windows, actual_size):
        positions = []
        sizes = []
        num_full_strides = num_windows - 1

        # For all windows except the last one
        for i in range(num_full_strides):
            positions.append(i * stride)
            sizes.append(window_size)

        # Handle the last window to ensure coverage of actual data size
        last_start = actual_size - window_size
        positions.append(last_start)
        sizes.append(window_size)

        return positions, sizes

    # 6 windows in latitude direction (5 strides of 50 + last window)
    lat_positions, lat_sizes = get_windows(6, LAT_SIZE)

    # 12 windows in longitude direction (11 strides of 50 + last window)
    lon_positions, lon_sizes = get_windows(13, LON_SIZE)

    # print(f"[INFO] {current_time()} Latitude positions: {lat_positions}")
    # print(f"[INFO] {current_time()} Longitude positions: {lon_positions}")

    return lat_positions, lon_positions, lat_sizes, lon_sizes

def process_subgraph(df, subregion_row, subregion_col, lat_start, lon_start, lat_size, lon_size, itr_no):
    """
    Process a subgraph within a specified window of the lunar region.

    Args:
        df (pd.DataFrame): Full subregion dataframe
        subregion_row (int): Subregion row index
        subregion_col (int): Subregion column index
        lat_start (int): Starting latitude index
        lon_start (int): Starting longitude index
        lat_size (int): Latitude window size
        lon_size (int): Longitude window size

    Returns:
        str: Path where the graph is saved
    """
    # global OccupancyMatrix

    # Calculate indices for the subgraph
    indices = []
    for i in range(lat_size):
        row_start = (lat_start + i) * LAT_STEP + lon_start
        indices.extend(range(row_start, row_start + lon_size))

    row_idx = int(np.ceil(float(lat_start)/STRIDE))
    col_idx = int(np.ceil(float(lon_start)/STRIDE))
    print(f"[INFO] {current_time()} Started processing subregion {subregion_row} {subregion_col}, subgraph id: {row_idx} {col_idx}")

    # Subset the dataframe
    sub_df = df.iloc[indices].copy()

    # Extract features for the subgraph
    latitudes = sub_df['lat_center'].values
    longitudes = sub_df['lon_center'].values
    mareOrHighland = sub_df['mareOrHighland'].values
    w_vectors = sub_df[[f'w_{i}' for i in range(1, wCount + 1)]].values
    element_columns = elements
    element_compositions = sub_df[element_columns].values
    if itr_no == 1:
        updates = sub_df['updated'].values
        updates_tensor = torch.tensor(updates, dtype=torch.bool).to(device)

    # non_zero_rows = np.sum((sub_df[element_columns].sum(axis=1) > 0).values)
    # total_rows = len(sub_df)
    # percentage_non_zero = (non_zero_rows / total_rows) * 100
    # OccupancyMatrix[row_idx, col_idx] = percentage_non_zero

    # Move data to GPU and preprocess
    latitudes_tensor = torch.tensor(latitudes, dtype=torch.float32).to(device)
    longitudes_tensor = torch.tensor(longitudes, dtype=torch.float32).to(device)
    mareOrHighland_tensor = torch.tensor(mareOrHighland, dtype=int).to(device)

    scaler_w = StandardScaler()
    w_vectors_tensor = torch.tensor(scaler_w.fit_transform(w_vectors), dtype=torch.float32).to(device)
    element_compositions_tensor = torch.tensor(element_compositions, dtype=torch.float32).to(device)

    # if (itr_no == 1):
    #     # Need to compute element mask only in the first iteration, otherwise load it
    #     element_mask_tensor = torch.tensor((sub_df[element_columns].sum(axis=1) > 0).values, dtype=torch.bool).to(device)
    #     os.makedirs(f'./drive/MyDrive/ISRO_SuperResolution/masks/masks_subregion_{subregion_row}_{subregion_col}', exist_ok=True)
    #     torch.save(element_mask_tensor, f'./drive/MyDrive/ISRO_SuperResolution/masks/masks_subregion_{subregion_row}_{subregion_col}/mask_tensor_{subregion_row}_{subregion_col}_{row_idx}_{col_idx}.pt')
    #     print(f"[INFO] {current_time()} Created mask for subgraph: {row_idx}, {col_idx}")
    # else :
    #     element_mask_tensor = torch.load(f'masks/masks_subregion_{subregion_row}_{subregion_col}/mask_tensor_{subregion_row}_{subregion_col}_{row_idx}_{col_idx}.pt', weights_only = True)
    #     print(f"[INFO] {current_time()} Loaded mask for subgraph : {row_idx}, {col_idx}")

    os.makedirs(f'masks/masks_subregion_{subregion_row}_{subregion_col}', exist_ok=True)
    if os.path.isfile(f'masks/masks_subregion_{subregion_row}_{subregion_col}/mask_tensor_{subregion_row}_{subregion_col}_{row_idx}_{col_idx}.pt'):
        # So mask already exists, need to update if iteration is 1
        element_mask_tensor = torch.load(f'masks/masks_subregion_{subregion_row}_{subregion_col}/mask_tensor_{subregion_row}_{subregion_col}_{row_idx}_{col_idx}.pt', weights_only = True)
        # I need to update the tensor using the updated column in the dataframe
        print(f"[INFO] {current_time()} Loaded mask for subgraph : {row_idx}, {col_idx}")
        if itr_no == 1:
            element_mask_tensor = torch.logical_or(element_mask_tensor, updates_tensor)
            torch.save(element_mask_tensor, f'masks/masks_subregion_{subregion_row}_{subregion_col}/mask_tensor_{subregion_row}_{subregion_col}_{row_idx}_{col_idx}.pt')
            print(f"[INFO] {current_time()} Updated and saved new mask for subgraph : {row_idx}, {col_idx} in iteration {itr_no}")
    else:
        # Need to compute element mask if it does not exist
        # Mask only needs to be computed in the first iteration only using updates tensor
        if itr_no == 1:
            element_mask_tensor = updates_tensor
            # element_mask_tensor = torch.tensor((sub_df[element_columns].sum(axis=1) > 0).values, dtype=torch.bool).to(device)
            os.makedirs(f'masks/masks_subregion_{subregion_row}_{subregion_col}', exist_ok=True)
            torch.save(element_mask_tensor, f'masks/masks_subregion_{subregion_row}_{subregion_col}/mask_tensor_{subregion_row}_{subregion_col}_{row_idx}_{col_idx}.pt')
            print(f"[INFO] {current_time()} Created and saved mask for subgraph: {row_idx}, {col_idx}")
        else:
            # So for this subregion mask was not created in the first iteration and it does not even exist
            # Thus the entire mask will be zero, it will just be a temporary mask and will not be saved
            element_mask_tensor = torch.tensor(np.zeros(len(sub_df)), dtype=torch.bool).to(device)
            print(f"[INFO] {current_time()} Created temporary zero mask for subgraph: {row_idx}, {col_idx}")


    # print(f"[INFO] {current_time()} Extracted features from dataframe, Subgraph id: {int(np.ceil(float(lat_start)/STRIDE))} {int(np.ceil(float(lon_start)/STRIDE))}")

    # Process graph edges
    edge_index = []
    edge_weights = []
    num_nodes = len(latitudes)

    # Iterate over batches to compute edge weights efficiently
    for batch_start in range(0, num_nodes, BATCH_SIZE):
        # print(f"[INFO] {current_time()} Starting batch {batch_start}, Subgraph id: {int(np.ceil(float(lat_start)/STRIDE))} {int(np.ceil(float(lon_start)/STRIDE))}")

        batch_lat = latitudes_tensor[batch_start:batch_start+BATCH_SIZE]
        batch_lon = longitudes_tensor[batch_start:batch_start+BATCH_SIZE]
        batch_w = w_vectors_tensor[batch_start:batch_start+BATCH_SIZE]
        batch_mare = mareOrHighland_tensor[batch_start: batch_start+BATCH_SIZE]

        # print(f"[DEBUG] batch_lat shape: {batch_lat.shape}")
        # print(f"[DEBUG] batch_lon shape: {batch_lon.shape}")
        # print(f"[DEBUG] batch_w shape: {batch_w.shape}")

        # Compute distances
        Dspatial = compute_spatial_distances_gpu(batch_lat, batch_lon, latitudes_tensor, longitudes_tensor)
        Dfeature = compute_feature_distances_gpu(batch_w, w_vectors_tensor)
        Dmare = compute_mare_highland_distances_gpu(batch_mare, mareOrHighland_tensor)
        weights = compute_edge_weights_gpu(Dspatial, Dfeature, Dmare, ALPHA, BETA, GAMMA)
        
        # Debugging: Check distance matrix shapes
        # print(f"[DEBUG] Dspatial shape: {Dspatial.shape}")
        # print(f"[DEBUG] Dfeature shape: {Dfeature.shape}")
        # print(f"[DEBUG] weights shape: {weights.shape}")
        # print(f"[INFO] {current_time()} Computed distances {batch_start}, Subgraph id: {int(np.ceil(float(lat_start)/STRIDE))} {int(np.ceil(float(lon_start)/STRIDE))}")

        # Select top-k neighbors for all nodes in the batch at once
        top_k_values, top_k_indices = torch.topk(weights, K+1, dim=1, largest=True)
        top_k_values = top_k_values[:,1:]
        top_k_indices = top_k_indices[:,1:]

        # Flatten and append edge indices and weights in one go
        batch_indices = torch.arange(batch_start, batch_start + weights.size(0), device=weights.device).unsqueeze(1).repeat(1, K+1).flatten()
        top_k_indices_flat = top_k_indices.flatten()
        top_k_values_flat = top_k_values.flatten()

        # Append edge indices and weights in a vectorized manner
        edge_index.extend(zip(batch_indices.tolist(), top_k_indices_flat.tolist()))
        edge_weights.extend(top_k_values_flat.tolist())

    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

    # Prepare node features
    node_features = torch.hstack((
        w_vectors_tensor,
        mareOrHighland_tensor.reshape(-1, 1),
        latitudes_tensor.reshape(-1, 1),
        longitudes_tensor.reshape(-1, 1),
    ))

    # Create PyTorch Geometric Data object
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_weights,
        y=element_compositions_tensor,
        mask=element_mask_tensor,
    )

    # Add metadata for tracking and analysis
    graph.metadata = {
        "node_features_shape": node_features.shape,
        "edge_index_shape": edge_index.shape,
        "edge_attr_shape": edge_weights.shape,
        "element_compositions_shape": element_compositions_tensor.shape,
        "element_mask_shape": element_mask_tensor.shape,
        "subregion_indices": {
            "row": subregion_row,
            "col": subregion_col,
            "subgraph_row": row_idx,
            "subgraph_col": col_idx
        }
    }

    # Save graph
    graphs_dir = f"./graphs/graphs_subregion_{subregion_row}_{subregion_col}"
    os.makedirs("./graphs/", exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # Define save path using new directory structure
    save_path = os.path.join(
        graphs_dir,
        f"subgraph_{subregion_row}_{subregion_col}_{row_idx}_{col_idx}.pt"
    )

    # Save graph
    torch.save(graph, save_path)

    print(f"[INFO] {current_time()} Saved subgraph to {save_path}")

    return save_path


def Part3(subregion_row, subregion_col, iteration_number, updatedIndices, updatedSubregions):
    # updatedIndices is a list of csv indices where abundances are added by the file being currently processed
    # updatedSubregions is an array of size 6x13 denoting number of updates in each subregion
    # In iteration 1, we do not need to generate graphs for subregions where updatedSubregions is 0 and also we need to update/generate the mask

    # Construct filename and verify file exists
    fileName = f"./regions/ISRO_RegionData{subregion_row - subregion_row%2}{1+subregion_row - subregion_row%2}/subregion_{subregion_row}_{subregion_col}.csv"
    if not os.path.isfile(fileName):
        print(f"[ERROR] File (subregion_{subregion_row}_{subregion_col}.csv) does not exist. Exiting...")
        exit()

    # Read CSV file
    df = pd.read_csv(fileName)
    print(f"[INFO] {current_time()} Dataframe Read. Size = {df.memory_usage(deep=True).sum()/(1024*1024):6f} MB")

    lat_positions, lon_positions, lat_sizes, lon_sizes = calculate_sliding_windows()

    # Add column for updating mask with updated indexes
    if iteration_number == 1:
        df['updated'] = np.where(df.index.isin(updatedIndices), 1, 0)

    # Compute total number of subgraphs
    total_graphs = len(lat_positions) * len(lon_positions)
    print(f"[INFO] {current_time()} Will generate {total_graphs} subgraphs ({len(lat_positions)} rows x {len(lon_positions)} columns)")
    # Define thread pool and process data
    with ThreadPoolExecutor() as executor:
        for j_index, lon_start in enumerate(lon_positions):
            futures = []
            for i_index, lat_start in enumerate(lat_positions):
                if iteration_number == 1:
                    if updatedSubregions[i_index][j_index] == 0:
                        print(f"Skipping subgraph {i_index} {j_index} for Region {subregion_row}{subregion_col}: No updated entries")
                        continue
                # Submit each task to the thread pool
                print(f"[INFO] {current_time()} Generating subgraph {i_index} {j_index} for Region {subregion_row} {subregion_col}")
                futures.append(executor.submit(process_subgraph, df, subregion_row, subregion_col, lat_start, lon_start, lat_sizes[i_index], lon_sizes[j_index], iteration_number))

            for future in futures:
                future.result()

# # Part 4: Train all Subgraphs
# This exposes a function (Part4), which given any i, j, iteration_number will load and train the combined model in an advanced mini batch fashion on all the 78 subgraphs and also delete those now redundant subgraphs and update the csv file with the calculated abundances.

num_targets = len(elements)


class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads = 1, edge_dim = 1)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads = 1, edge_dim = 1)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x


class CNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        return x


class CombinedModel(nn.Module):
    def __init__(
        self, gnn_in_channels, gnn_hidden_channels, gnn_out_channels,
        cnn_in_channels, cnn_out_channels, fusion_hidden_channels, num_elements
    ):
        super(CombinedModel, self).__init__()
        self.gnn = GNNModel(gnn_in_channels, gnn_hidden_channels, gnn_out_channels)
        self.cnn = CNNModel(cnn_in_channels, cnn_out_channels)
        self.fc1 = nn.Linear(gnn_out_channels + cnn_out_channels, fusion_hidden_channels)
        self.fc2 = nn.Linear(fusion_hidden_channels, num_elements)

    def forward(self, data):
        # GNN forward
        x_gnn = self.gnn(data.x, data.edge_index, data.edge_attr)

        grid_size = (100, 100)
        grid = torch.zeros((1, data.x.size(1), grid_size[0], grid_size[1]), device = data.x.device)
        coords = data.x[:, -2:]
        latitudes, longitudes = coords[:, 0], coords[:, 1]

        lat_indices = ((latitudes - latitudes.max()) / (latitudes.min() - latitudes.max()) * (grid_size[0] - 1)).int()
        lon_indices = ((longitudes - longitudes.min()) / (longitudes.max() - longitudes.min()) * (grid_size[1] - 1)).int()

        lat_indices = torch.clamp(lat_indices, 0, grid_size[0] - 1)
        lon_indices = torch.clamp(lon_indices, 0, grid_size[1] - 1)

        grid[0, :, lat_indices, lon_indices] = data.x.t()
        x_cnn = self.cnn(grid)

        # Compute lat_indices and lon_indices directly for the reduced grid
        lat_indices = ((latitudes - latitudes.max()) / (latitudes.min() - latitudes.max()) * (x_cnn.shape[2] - 1)).int()
        lon_indices = ((longitudes - longitudes.min()) / (longitudes.max() - longitudes.min()) * (x_cnn.shape[3] - 1)).int()
        lat_indices = torch.clamp(lat_indices, 0, x_cnn.shape[2] - 1)
        lon_indices = torch.clamp(lon_indices, 0, x_cnn.shape[3] - 1)

        x_cnn = x_cnn[0, :, lat_indices, lon_indices].t()

        # Combine GNN and CNN outputs
        x_combined = torch.cat((x_gnn, x_cnn), dim=1)
        x = F.relu(self.fc1(x_combined))
        x = self.fc2(x)
        x = F.relu(x)

        return x.squeeze()


# TASK 4: Loss Function
mse = nn.MSELoss(reduction='none')
def masked_mse_loss(predictions, targets, mask, element_weights):
    """
    Computes Weighted MSE loss only where the mask is True.

    Args:
        predictions: Predicted values of shape (num_nodes, 8).
        targets: Ground truth values of shape (num_nodes, 8).
        mask: Boolean mask indicating known values of shape (num_nodes, 8).

    Returns:
        Weighted MSE loss computed only for the known (masked) values.
    """
    loss = mse(predictions, targets)*element_weights
    masked_loss = loss * mask.unsqueeze(1)  # Apply mask
    return masked_loss.sum() / mask.sum()

def feature_similarity_loss(predictions, features, edge_index):
    """
    Penalizes abundance differences for nodes with similar features.

    Args:
        predictions: Predicted abundances (num_nodes, num_elements).
        features: Node feature vectors (num_nodes, feature_dim).
        edge_index: Edge indices (2, num_edges).

    Returns:
        Feature similarity loss.
    """
    src, dest = edge_index  # Source and destination nodes
    diff_predictions = predictions[src] - predictions[dest]  # Abundance differences
    diff_features = features[src] - features[dest]  # Feature differences
    # normalized_diff_feature = (diff_features - diff_features.min()) / (diff_features.max() - diff_features.min() + 1e-8)  # Avoid division by zero

    # Normalize per edge, not globally across the whole tensor
    diff_features_norm = torch.norm(diff_features, dim=1, keepdim=True)
    normalized_diff_feature = diff_features / (diff_features_norm + 1e-8)

    weights = torch.exp(-torch.norm(normalized_diff_feature, dim=1))  # Similarity weight (higher for similar features)
    return (weights.unsqueeze(1) * diff_predictions**2).mean()

def logarithmic_loss(predictions, targets, mask, element_weights):
    """
    Computes the logarithmic loss with masking and element-wise weighting.

    Parameters:
    - predictions (torch.Tensor): Predicted values (batch_size x num_elements).
    - targets (torch.Tensor): True values (batch_size x num_elements).
    - mask (torch.Tensor): Binary mask (batch_size,) to indicate valid samples.
    - element_weights (torch.Tensor, optional): Weights for each element (num_elements,).

    Returns:
    - torch.Tensor: Computed loss (scalar).
    """
    epsilon = 1e-8
    log_diff = (torch.log(predictions + epsilon) - torch.log(targets + epsilon))
    log_loss = (log_diff ** 2) * element_weights  # Square the difference
    masked_log_loss = log_loss*mask.unsqueeze(1)
    return (masked_log_loss.sum()) / (mask.sum() + epsilon)    

def spatial_similarity_loss(predictions, edge_index, lat_lon):
    """
    Penalizes abundance differences for spatially close nodes using Euclidean distance.

    Args:
        predictions: Predicted abundances (num_nodes, num_elements).
        edge_index: Edge indices (2, num_edges).
        lat_lon: Tensor containing latitudes and longitudes for each node (num_nodes, 2).

    Returns:
        Spatial similarity loss.
    """

    src, dest = edge_index  # Source and destination nodes
    lat_lon_src = lat_lon[src]  # Get latitudes and longitudes for source nodes
    lat_lon_dest = lat_lon[dest]  # Get latitudes and longitudes for destination nodes

    distance = torch.norm(lat_lon_src - lat_lon_dest, dim=1)
    normalized_distance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-8)  # Avoid division by zero

    diff_predictions = predictions[src] - predictions[dest]  # (num_edges, num_elements)
    weights = torch.exp(-normalized_distance)
    spatial_loss = (weights.unsqueeze(1) * diff_predictions**2).mean()  # (num_edges, num_elements)

    return spatial_loss


def combined_loss(mode, predictions, targets, mask, w_features, edge_index, element_weights, lat_lon, metadata, lambda_reg=0.1):
    # Calculated only for known points for subgraphs where mask is present
    # subregion_row_loss = metadata['subregion_indices']['row']
    # subregion_col_loss = metadata['subregion_indices']['col']
    # subregion_srow = metadata['subregion_indices']['subgraph_row']
    # subregion_scol = metadata['subregion_indices']['subgraph_col']

    if mode == 1:
        mse_loss = masked_mse_loss(predictions, targets, mask, element_weights)
        # log_loss = logarithmic_loss(predictions, targets, mask, element_weights)

        return mse_loss
    elif mode == 2:
        feature_sim_loss = feature_similarity_loss(predictions, w_features, edge_index)
        spatial_loss = spatial_similarity_loss(predictions, edge_index, lat_lon)
        mse_loss = masked_mse_loss(predictions, targets, mask, element_weights)
        log_loss = logarithmic_loss(predictions, targets, mask, element_weights)

        return mse_loss + lambda_reg*log_loss + feature_sim_loss + spatial_loss
    elif mode == 3:
        feature_sim_loss = feature_similarity_loss(predictions, w_features, edge_index)
        spatial_loss = spatial_similarity_loss(predictions, edge_index, lat_lon)
    
        return feature_sim_loss + spatial_loss
    else:
        return None

    # # Calculated for all points
    # feature_sim_loss = feature_similarity_loss(predictions, w_features, edge_index)
    # spatial_loss = spatial_similarity_loss(predictions, edge_index, lat_lon)

    # if os.path.isfile(f"./drive/MyDrive/ISRO_SuperResolution/masks/masks_subregion_{subregion_row_loss}_{subregion_col_loss}/mask_tensor_{subregion_row_loss}_{subregion_col_loss}_{subregion_srow}_{subregion_scol}.pt"):
    #     mse_loss = masked_mse_loss(predictions, targets, mask, element_weights)
    #     log_loss = logarithmic_loss(predictions, targets, mask, element_weights)
    #     print(feature_sim_loss.item(), spatial_loss.item(), mse_loss.item(), log_loss.item())
    #     return mse_loss + lambda_reg*log_loss + lambda_reg*feature_sim_loss + lambda_reg*spatial_loss
    #     # return mse_loss + lambda_reg*log_loss
    

class EarlyStopper:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_loss = None

    def early_stop(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss > loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False    

def save_model(epoch, checkpoint_dir, model):
    model_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] {current_time()} Model saved at {model_save_path}")

def train_combined(data, model, optimizer, epochs, element_weights, lr):
    model.train()
    torch.backends.cudnn.benchmark = True  # For consistent GPU performance
    torch.backends.cudnn.deterministic = False  # Slightly faster, less reproducible
    early = EarlyStopper()

    # I will use a dual training approach,
    # if the mask file is present, i.e. the subregion has some known enteries
    # Then initially I am only going to optimise the mse loss (mode = 1)
    # After which I am going to train on all the 4 losses (mode = 2)
    # If no mask is present, then trained only using the first 2 losses (mode = 3)
    # Thus the loss calculation has a parameter mode = (1, 2, 3) and returns a list of Pytorch loss objects

    subregion_row_loss = data.metadata['subregion_indices']['row']
    subregion_col_loss = data.metadata['subregion_indices']['col']
    subregion_srow = data.metadata['subregion_indices']['subgraph_row']
    subregion_scol = data.metadata['subregion_indices']['subgraph_col']
    totalEpochs = int(epochs*(3/2))

    if os.path.isfile(f"./drive/MyDrive/ISRO_SuperResolution/masks/masks_subregion_{subregion_row_loss}_{subregion_col_loss}/mask_tensor_{subregion_row_loss}_{subregion_col_loss}_{subregion_srow}_{subregion_scol}.pt"):
        # Mask is present, train for epochs using mse loss
        with tqdm(range(totalEpochs), desc="Training... (mask +nt)") as pbar:
            mode = 1
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, ((totalEpochs//2)+1), eta_min=0.0001)

            for epoch in range(totalEpochs//2):
                optimizer.zero_grad(set_to_none=True)
                output = model(data)

                loss = combined_loss(mode, output, data.y, data.mask, data.x[:, :300], data.edge_index, element_weights, data.x[:, -2:], data.metadata)
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                pbar.set_postfix_str(f"{loss.item():4f}")
                pbar.update(1)

                if early.early_stop(loss.item()):
                    break

                if epoch != (epochs - 2):
                    scheduler.step()
                
                torch.cuda.empty_cache()

            mode = 2
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, (totalEpochs - (epochs//2) + 1), eta_min=0.0001)
            
            for epoch in range(totalEpochs - (totalEpochs//2)):
                optimizer.zero_grad(set_to_none=True)
                output = model(data)

                loss = combined_loss(mode, output, data.y, data.mask, data.x[:, :300], data.edge_index, element_weights, data.x[:, -2:], data.metadata)
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                pbar.set_postfix_str(f"{loss.item():4f}")
                pbar.update(1)

                if epoch != (totalEpochs-2):
                    scheduler.step()

                torch.cuda.empty_cache()
    else:
        mode = 3
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, (epochs + 1), eta_min=0.0001)

        with tqdm(range(epochs), desc="Training... (mask -nt)") as pbar:
            for epoch in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                output = model(data)

                loss = combined_loss(mode, output, data.y, data.mask, data.x[:, :300], data.edge_index, element_weights, data.x[:, -2:], data.metadata)
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                pbar.set_postfix_str(f"{loss.item():4f}")
                pbar.update(1)

                if early.early_stop(loss.item()):
                    break

                if epoch != (epochs-2):
                    scheduler.step()

                torch.cuda.empty_cache()

     # print(f'[INFO] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')


# Hyperparameters
gnn_in, gnn_hidden, gnn_out = 303, 128, 36
cnn_in, cnn_out = 303, 36
fusion_hidden, num_targets = 64, num_targets
element_weights = torch.tensor([2, 1, 1, 3, 3, 3, 1, 1]).to(device)


def train_graph(graph_data, model, optimizer, num_epochs = 100, lr = 0.001):
    graph_data = graph_data.to(device)
    train_combined(data=graph_data, model=model, optimizer=optimizer, epochs=num_epochs, element_weights= element_weights, lr = lr)


def find_grid_indices_in_subregion2(lat, lon, subregion_row, subregion_col):
    """
    Calculate the grid indices for a given latitude and longitude within a subregion.

    Parameters:
    - lat: Latitude of the point.
    - lon: Longitude of the point.
    - subregion_row: Row index of the subregion.
    - subregion_col: Column index of the subregion.

    Returns:
    - Tuple (square_row, square_col) representing the grid indices.
    """
    # Calculate the start of the subregion
    subregion_lat_start = LATITUDE_RANGE[0] - subregion_row * lat_per_region
    subregion_lon_start = LONGITUDE_RANGE[0] + subregion_col * lon_per_region

    # Calculate the grid square within the subregion
    square_row = abs(int((lat - subregion_lat_start) // square_size_deg))
    square_col = abs(int((lon - subregion_lon_start) // square_size_deg))

    if not (0 <= square_row < num_squares_lat) or not (0 <= square_col < num_squares_lon):
        # If point is out of the subregion's grid, find the nearest grid point
        square_row = max(0, min(square_row, num_squares_lat - 1))
        square_col = max(0, min(square_col, num_squares_lon - 1))

    # Calculate the exact row index
    row_index = square_row * num_squares_lon + square_col

    return row_index

def find_grid_indices_in_subregion2_vectorized(latitudes, longitudes, subregion_row, subregion_col):
    """
    Vectorized calculation of grid indices for given latitudes and longitudes within a subregion using PyTorch.

    Parameters:
    - latitudes: Numpy Array of latitudes.
    - longitudes: Numpy Array of longitudes.
    - subregion_row: Row index of the subregion.
    - subregion_col: Column index of the subregion.

    Returns:
    - Numpy Array of row indices for the grid positions.
    """

    # Calculate the start of the subregion
    subregion_lat_start = LATITUDE_RANGE[0] - subregion_row * lat_per_region
    subregion_lon_start = LONGITUDE_RANGE[0] + subregion_col * lon_per_region

    # # Calculate the grid squares within the subregion
    # square_rows = torch.abs(((latitudes - subregion_lat_start) // square_size_deg).to(torch.int))
    # square_cols = torch.abs(((longitudes - subregion_lon_start) // square_size_deg).to(torch.int))

    # Calculate the grid squares within the subregion
    square_rows = np.abs(((latitudes - subregion_lat_start) // square_size_deg).astype(int))
    square_cols = np.abs(((longitudes - subregion_lon_start) // square_size_deg).astype(int))

    # # Clamp indices to valid grid ranges
    # square_rows = torch.clamp(square_rows, 0, num_squares_lat - 1)
    # square_cols = torch.clamp(square_cols, 0, num_squares_lon - 1)

    # Clamp indices to valid grid ranges
    square_rows = np.clip(square_rows, 0, num_squares_lat - 1)
    square_cols = np.clip(square_cols, 0, num_squares_lon - 1)

    # Calculate the exact row indices
    row_indices = square_rows * num_squares_lon + square_cols

    return row_indices

def save_predictions(d, o, graph_data, graph_output):
    """
    Optimized save_predictions function using vectorized grid index computation.
    This function takes as argument a graph_data object and iterates through all rows of the data.
    It uses the index to get the abundance vector from the model output on the graph_data.
    It then figures out the index position in the csv file
    Then saves the data if the csv file is initially empty
    Then closes the file at the end.
    """

    # Read the CSV once
    df_path = f'./regions/RegionData{d - d%2}{1 + d - d%2}/subregion_{d}_{o}.csv'
    df = pd.read_csv(df_path)

    metadata = graph_data.metadata
    subregion_row = metadata['subregion_indices']['row']
    subregion_col = metadata['subregion_indices']['col']
    w = 0.25

    assert subregion_row == d, "[ERROR] Metadata row does not match d"
    assert subregion_col == o, "[ERROR] Metadata col does not match o"

    # Extract latitudes, longitudes, mask, and predictions as numpy arrays
    latitudes = graph_data.x[:, -2].cpu().numpy()
    longitudes = graph_data.x[:, -1].cpu().numpy()
    masks = graph_data.mask.cpu().numpy()
    predictions = graph_output.cpu().numpy()
    df_indices = find_grid_indices_in_subregion2_vectorized(
        latitudes, 
        longitudes, 
        subregion_row, 
        subregion_col
    )

    # Filter valid rows (where data needs to be imputed) where mask is 0
    valid_rows = masks == 0
    df_indices = df_indices[valid_rows]
    predictions = predictions[valid_rows]

    # Extract the relevant rows from the DataFrame for processing
    df_elements = df.loc[df_indices, elements].values

    # Create a mask to identify rows with non-zero elements
    nonzero_mask = (df_elements != 0).any(axis=1)

    # Weighted update for rows with non-zero elements
    df_elements[nonzero_mask] = (w * df_elements[nonzero_mask]) + ((1 - w) * predictions[nonzero_mask])

    # Direct update for rows with all-zero elements
    df_elements[~nonzero_mask] = predictions[~nonzero_mask]

    # Write the updated values back to the DataFrame
    df.loc[df_indices, elements] = df_elements

    # # Apply updates to the DataFrame
    # for idx, prediction in zip(df_indices.tolist(), predictions.cpu().tolist()):
    #     if (df.loc[idx, elements] != 0).any():
    #         # Weighted update
    #         df.loc[idx, elements] = (w * df.loc[idx, elements]) + ((1 - w) * prediction)
    #     else:
    #         # Direct update
    #         df.loc[idx, elements] = prediction

    # Save the updated DataFrame
    df.to_csv(df_path, index=False)
    print(f"[INFO] {current_time()} Saved predictions to {df_path} for subregion {graph_data.metadata['subregion_indices']['subgraph_row']} {graph_data.metadata['subregion_indices']['subgraph_col']}")


Part4executor = ThreadPoolExecutor(max_workers=1)  # Adjust max_workers based on available resources

def Part4(x, y, iteration_number):
    pattern = r"subgraph_(\d+)_(\d+)_(\d+)_(\d+)\.pt"

    # I need to iterate through all the x, y files in the directory graphs/subregion_i_j/subgraph_i_j_x_y.pt
    directory = f"./graphs/graphs_subregion_{x}_{y}/"
    model = CombinedModel(gnn_in, gnn_hidden, gnn_out, cnn_in, cnn_out, fusion_hidden, num_targets).to(device)

    os.makedirs(f'./drive/MyDrive/ISRO_SuperResolution/models', exist_ok=True)
    if os.path.isfile(f'./drive/MyDrive/ISRO_SuperResolution/models/{x}_{y}.pth'):
        model.load_state_dict(torch.load(f"./drive/MyDrive/ISRO_SuperResolution/models/{x}_{y}.pth"))
        num_epochs = 300 # Only need to finetune later
        lr = 0.0005
    else:
        num_epochs = 700
        lr = 0.001

    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    optimizer = optim.Yogi(model.parameters(), lr=lr, weight_decay=1e-3)
    
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            torch.cuda.empty_cache()
            file_path = os.path.join(directory, filename)

            # Extract i, j, x, and y using the regular expression
            match = re.match(pattern, filename)
            if not match:
                print(f"[ERROR] Filename {filename} does not match expected pattern. Skipping.")
                continue

            i2, j2, x1, y1 = map(int, match.groups())
            x1 = int(x1)
            y1 = int(y1)
            i2 = int(i2)
            j2 = int(j2)
            del match

            print(f"[INFO] {current_time()} Starting", filename)

            assert i2 == x, "[ERROR] File Number row index do not match"
            assert j2 == y, "[ERROR] File Number col index do not match"

            # Now I need to do the following, if iteration number is 1. Then I need to ignore those subgraphs (x, y)
            # s.t. the amount of data in that subgraph is less than 10%
            # This data is stored in the following pickle file: ./density/OccupanyMatrix_i_j.pkl
            # which is loaded in the Occupancy Variable
            # if iteration_number == 1:
            #     if(Occupancy[0, 0] < 10):
            #         continue
            # save_directory = f'{checkpoint_dir}/subregion_{x}_{y}/{iteration_number}/{x1}_{y1}/'
            # os.makedirs(save_directory, exist_ok=True)

            graph_data = torch.load(file_path)
            # For each data object file I need to train the Graph using that data
            train_graph(graph_data, model, optimizer, num_epochs, lr)

            print(f"[INFO] {current_time()} Trained model on {filename}. Now saving predictions.")

            # # Finally for that graph I need to open the region csv file and place where data is not avalaible
            # save_predictions(x, y, model, graph_data)

            # Compute the output on the CPU and pass it to save_predictions
            graph_output = model(graph_data).cpu().detach()
            _ = Part4executor.submit(save_predictions, x, y, graph_data, graph_output)

            os.remove(file_path)

            print(f"[INFO] {current_time()} Processed and deleted {filename}")

            torch.cuda.empty_cache()
            torch.save(model.state_dict(), f"/models/{x}_{y}.pth")

 
# # Part 5: Final
# This part uses Part3 and Part4 function in an iterative loop to get the final abundances csv. It also does some final processing to make it easier for mapping.
def HandleRegion(i, j, num_iterations, updatedIndices, updatedSubregions):
  """
  This function is responsible for handling all update compuations of a region (out of the 64 regions) indexed by i, j
  """
  for iteration in range(1, num_iterations + 1):
    print(f"[INFO] {current_time()} Starting Iteration {iteration} for region {i} {j}")

    print(f"[INFO] {current_time()} Running PART 3 (iter: {iteration})")
    Part3(i, j, iteration, updatedIndices, updatedSubregions) 
    # Note that updatedIndices is a list of indices in the csv
    # updatedSubregions is a 6x13 matrix of number of enteries added in each subregion

    print(f"[INFO] {current_time()} Running PART 4 (iter: {iteration})")
    Part4(i, j, iteration)

from concurrent.futures import ThreadPoolExecutor

# toProcess = np.array([(0,2), (0,3), (0,4), (0,5), (0,6), (3,0), (3,1), (3,2), (3,3)])

# Final function
def ProcessDataP1(file_name, num_iteration):
  """
  This will implement the dynamic nature required
  Arguments:
    - df:
        - rows: observations
        - headers: lat0, lon0, lat1, lon1, lat2, lon2, lat3, lon3, 'Fe', 'Ti', 'Ca', 'Si', 'Al', 'Mg', 'Na', 'O'
    - num_iterations:
        - Number of iterations to finetune the output on

  Functionality:
    - Updates the mapping csv file with the new data
  """

  """
  Directory Structure Assumptions:
    - root
        | - regions/subregion_i_j.csv, for all i, j in [1 ... 8]: This is the global information bank which is continously updated
        | - LROC_GLOBAL_MARE_180.DBF
        | - LROC_GLOBAL_MARE_180.PRJ
        | - LROC_GLOBAL_MARE_180.SHP
        | - LROC_GLOBAL_MARE_180.SHP.XML
        | - LROC_GLOBAL_MARE_180.SHX
        | - LROC_GLOBAL_MARE_README.TXT
  """

  # First thing it will do is call PART2 function with arguments: dataframe
  # PART2 will update CSV files with the new enteries
  # and return number of new rows in each region (64 regions)
  # and for each region number of new data points in each subgraph
  # and also the row indices of the subregion_i_j.csv file where values are updated
  print(f"[INFO] {current_time()} Starting...")
  print(f"[INFO] {current_time()} Calling Part2 to update GIB")
  updatedRegions, updatedSubregions, indices = Part2(file_name)

  np.savez_compressed(f'PART2Output.npz', 
                      updatedSubregions=updatedSubregions, 
                      updatedRegions=updatedRegions, 
                      indices=indices)
  print(f"[INFO] {current_time()} Successfully updated GIB. UpdatedRegions: {updatedRegions}, UpdatedSubregions shape: {updatedSubregions.shape}, UpdatedIndices shape: {indices.shape}")

  return updatedRegions, updatedSubregions, indices

# def ProcessDataP2(num_iteration, updatedRegions, updatedSubregions, indices):
#   with ThreadPoolExecutor(max_workers=6) as executor:
#     for a in range(8):
#       for b in range(8):
#         if (updatedRegions[a][b] > 0) and ((a, b) in toProcess):
#           print(f"[INFO] {current_time()} Starting thread for {a} {b}")
#           executor.submit(HandleRegion, a, b, num_iteration, indices[a][b], updatedSubregions[a][b])


# updatedRegions, updatedSubregions, indices = ProcessDataP1('blablabla.csv', 3) 
# # Running the code
# PART 2 is complete
# from numpy import load as np_load_old
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# updatedRegions = np.load('PART2Output.npz')['updatedRegions']
# updatedSubregions = np.load('PART2Output.npz')['updatedSubregions']
# indices = np.load('PART2Output.npz')['indices']
def retrieve_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Subpixel resolution argument parser.")

    # Add the 'mode' argument
    parser.add_argument('mode', type=int, choices=[1, 2], help="Mode of operation (1 or 2).")

    # Arguments for mode 1
    parser.add_argument('filename', type=str, nargs='?', help="Filename (required if mode is 1).")

    # Arguments for mode 2
    parser.add_argument('i', type=int, nargs='?', help="Value of i (required if mode is 2).")
    parser.add_argument('j', type=int, nargs='?', help="Value of j (required if mode is 2).")
    parser.add_argument('num_iterations', type=int, nargs='?', help="Number of iterations (required if mode is 2).")
    args = parser.parse_args()

    if args.mode == 1:
        if args.filename is None:
            parser.error("Mode 1 requires the 'filename' argument.")
        print(f"Mode: {args.mode}, Filename: {args.filename}")
        return args.mode, args.filename

    elif args.mode == 2:
        if args.i is None or args.j is None or args.num_iterations is None:
            parser.error("Mode 2 requires 'i', 'j', and 'num_iterations' arguments.")
        print(f"Mode: {args.mode}, i: {args.i}, j: {args.j}, num_iterations: {args.num_iterations}")
        return args.mode, args.i, args.j, args.num_iterations
    
parsed = retrieve_arguments()

if parsed[0] == 1:
    # need to call part2
    updatedRegions, updatedSubregions, indices = ProcessDataP1(parsed[1], 3)
    print(updatedRegions)
elif parsed[0] == 2:
    # need to call handle region
    # Assumed part2 is called so check for file
    if os.path.isfile('PART2Output.npz'):
        from numpy import load as np_load_old
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        updatedRegions = np.load('PART2Output.npz')['updatedRegions']
        updatedSubregions = np.load('PART2Output.npz')['updatedSubregions']
        indices = np.load('PART2Output.npz')['indices']

        HandleRegion(parsed[1], parsed[2], parsed[3], indices[parsed[1]][parsed[2]], updatedSubregions[parsed[1]][parsed[2]])
    else:
        print("[ERROR] PART2Output.npz file not found. Please run mode 1 first. Exiting...")
        exit()