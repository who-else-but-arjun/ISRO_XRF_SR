import os
from astropy.io import fits
import numpy as np

# Define latitude and longitude divisions for 64 regions
LAT_DIVISIONS = [-90 + i * 22.5 for i in range(9)]  # Latitude (-90 to 90 divided into 8 parts)
LON_DIVISIONS = [-180 + i * 45 for i in range(9)]  # Longitude (-180 to 180 divided into 8 parts)

def determine_region(center_lat, center_lon):
    """
    Determine the 8x8 grid region index based on the center latitude and longitude.
    """
    for lat_idx in range(8):
        if LAT_DIVISIONS[lat_idx] <= center_lat < LAT_DIVISIONS[lat_idx + 1]:
            for lon_idx in range(8):
                if LON_DIVISIONS[lon_idx] <= center_lon < LON_DIVISIONS[lon_idx + 1]:
                    return lat_idx * 8 + lon_idx + 1  # 1-based region index
    return None

def process_fits_file(file_path):
    """
    Read a FITS file and extract necessary information: center coordinates and time range.
    """
    try:
        with fits.open(file_path) as hdul:
            header = hdul[1].header  # Assuming the relevant data is in extension 1
            # Extract latitudes and longitudes of the pixel corners
            latitudes = [header[f"V{i}_LAT"] for i in range(4)]
            longitudes = [header[f"V{i}_LON"] for i in range(4)]
            # Calculate center latitude and longitude
            center_lat = np.mean(latitudes)
            center_lon = np.mean(longitudes)

            # Extract start and end times
            start_time = header["STARTIME"]
            end_time = header["ENDTIME"]

            # Determine the region index
            region_index = determine_region(center_lat, center_lon)
            return region_index, start_time, end_time
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def traverse_directory(base_dir):
    """
    Traverse a directory structure (year/month/day) and process all FITS files.
    """
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue

        for month in os.listdir(year_path):
            month_path = os.path.join(year_path, month)
            if not os.path.isdir(month_path):
                continue

            for day in os.listdir(month_path):
                day_path = os.path.join(month_path, day)
                if not os.path.isdir(day_path):
                    continue

                for file_name in os.listdir(day_path):
                    if file_name.endswith(".fits"):
                        file_path = os.path.join(day_path, file_name)
                        region_index, start_time, end_time = process_fits_file(file_path)
                        if region_index:  # Ensure a valid region index
                            write_to_file(region_index, start_time, end_time)

def write_to_file(region_index, start_time, end_time):
    file_name = f"file_{region_index:02d}.txt"  # Zero-padded region index (e.g., file_01.txt)
    with open(file_name, "a") as file:
        file.write(f"{start_time} {end_time}\n")

if __name__ == "__main__":
    base_dir = "new/"  # Replace with the base directory path
    traverse_directory(base_dir)
    print("Everything done")

