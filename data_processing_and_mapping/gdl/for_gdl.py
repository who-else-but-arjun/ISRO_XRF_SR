from astropy.io import fits
from datetime import datetime, timedelta
import os

# --- Code 1: Convert GTI FITS file to UTC time intervals ---
# Path to the GTI FITS file
gti_file_path = "ch2_xsm_20230616_v1_level2.gti"

# Reference time for MET (Mission Elapsed Time)
met_reference = datetime(2017, 1, 1)

# Output file path for GTI times
gti_times_file = "gti_times_utc.txt"

# Open the FITS file
with fits.open(gti_file_path) as hdul:
    # Access the GTI HDU (assumed to be the second HDU)
    gti_hdu = hdul[1]
    start_times = gti_hdu.data['START']
    stop_times = gti_hdu.data['STOP']
    
    # Prepare the text file
    with open(gti_times_file, 'w') as f:
        for start, stop in zip(start_times, stop_times):
            # Convert MET to UTC
            start_utc = met_reference + timedelta(seconds=start)
            stop_utc = met_reference + timedelta(seconds=stop)
            
            # Format as yyyyMMddTHHmmssSSS
            start_str = start_utc.strftime('%Y%m%dT%H%M%S') + f"{start_utc.microsecond // 1000:03d}"
            stop_str = stop_utc.strftime('%Y%m%dT%H%M%S') + f"{stop_utc.microsecond // 1000:03d}"
            
            # Write to file
            f.write(f"{start_str} {stop_str}\n")

# --- Code 2: Match filenames to GTI intervals and sort them ---
# Paths
base_directory = "data/class/"  # Base path to the data/class directory
filenames_file = "filenames.txt"  # Output file to write matching filenames
sorted_filenames_file = "sorted_filenames.txt"  # File to write sorted filenames

# Helper function to check if a file falls in a given interval
def is_within_interval(filename, start_time, end_time):
    try:
        file_start_str, file_stop_str = filename.split("_")[3:5]
        file_start = datetime.strptime(file_start_str, '%Y%m%dT%H%M%S%f')
        file_stop = datetime.strptime(file_stop_str.split('.')[0], '%Y%m%dT%H%M%S%f')
    except (ValueError, IndexError):
        return False  # Skip invalid filenames
    
    # Check if the file's time interval overlaps with the GTI interval
    return not (file_stop < start_time or file_start > end_time)

# Read the GTI file and process each time interval
matching_files = []
with open(gti_times_file, 'r') as gti:
    for line in gti:
        start_str, stop_str = line.strip().split()
        start_time = datetime.strptime(start_str, '%Y%m%dT%H%M%S%f')
        stop_time = datetime.strptime(stop_str, '%Y%m%dT%H%M%S%f')
        
        # Construct the date directory
        date_dir = os.path.join(
            base_directory,
            start_time.strftime('%Y'),
            start_time.strftime('%m'),
            start_time.strftime('%d')
        ).replace(os.path.sep, '/')
        
        if not os.path.exists(date_dir):
            continue  # Skip if the directory doesn't exist
        
        # Scan the directory for FITS files
        for filename in os.listdir(date_dir):
            if filename.endswith(".fits") and is_within_interval(filename, start_time, stop_time):
                matching_files.append(filename)

# Write matching filenames to the output file
with open(filenames_file, 'w') as out:
    for file in matching_files:
        out.write(file + '\n')

# Read filenames from the matching files and sort them
with open(filenames_file, 'r') as file:
    filenames = [line.strip() for line in file.readlines()]

def extract_start_time(filename):
    try:
        start_time_str = filename.split("_")[3]
        return datetime.strptime(start_time_str, '%Y%m%dT%H%M%S%f')
    except (IndexError, ValueError):
        return None  # Return None for invalid filenames

sorted_filenames = sorted(filenames, key=extract_start_time)

# Write the sorted filenames to the output file
with open(sorted_filenames_file, 'w') as file:
    for filename in sorted_filenames:
        file.write(filename + '\n')

# --- Code 3: Generate GDL commands for continuous file batches ---
# Input and output for GDL commands
output_file = "command_gdl.txt"
input_directory_template = "data/class/{year}/{month}/{day}/"
output_directory = "combined"

# Helper function to parse timestamps from filenames
def parse_time(filename):
    try:
        start_time_str, end_time_str = filename.split("_")[3:5]
        start_time = datetime.strptime(start_time_str, '%Y%m%dT%H%M%S%f')
        end_time = datetime.strptime(end_time_str.split('.')[0], '%Y%m%dT%H%M%S%f')
        return start_time, end_time
    except (IndexError, ValueError):
        return None, None

# Read the sorted filenames
with open(sorted_filenames_file, 'r') as file:
    filenames = [line.strip() for line in file.readlines()]

# Output command file
with open(output_file, 'w') as command_file:
    i = 0
    while i <= len(filenames) - 12:
        batch = filenames[i:i + 12]
        is_continuous = True
        for j in range(len(batch) - 1):
            _, current_end = parse_time(batch[j])
            next_start, _ = parse_time(batch[j + 1])
            if current_end != next_start:
                is_continuous = False
                break
        
        if is_continuous:
            first_start, _ = parse_time(batch[0])
            _, last_end = parse_time(batch[-1])
            input_directory = input_directory_template.format(
                year=first_start.strftime('%Y'),
                month=first_start.strftime('%m'),
                day=first_start.strftime('%d'),
            )
            command = (
                f"CLASS_add_L1_files_time,"
                f"'{input_directory}',"
                f"'{first_start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}',"
                f"'{last_end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}',"
                f"'{output_directory}'\n"
            )
            command_file.write(command)
            i += 1  # Move to next batch
        else:
            i += batch.index(batch[j + 1])