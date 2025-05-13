import os
import pandas as pd
from datetime import datetime

def calculate_total_time_per_month(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse intervals and calculate duration for each
    monthly_durations = [0] * 12  # One entry for each month (0 = January, 11 = December)

    for line in lines:
        start_str, end_str = line.strip().split()
        start_time = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S.%f")
        end_time = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%S.%f")

        # Calculate duration in seconds
        duration = (end_time - start_time).total_seconds()

        # Add the duration to the corresponding month (month - 1 because index starts at 0)
        monthly_durations[start_time.month - 1] += duration

    return monthly_durations


# Prepare the 64x12 matrix
rows = 64  # Number of files
columns = 12  # One column for each month
matrix = []

# Process all files from file_01.txt to file_64.txt
for i in range(1, rows + 1):
    file_path = f"file_{i:02}.txt"  # File names like file_01.txt, file_02.txt, etc.
    if os.path.exists(file_path):
        monthly_durations = calculate_total_time_per_month(file_path)
    else:
        print(f"File {file_path} not found. Filling row with zeros.")
        monthly_durations = [0] * columns

    matrix.append(monthly_durations)

# Save the matrix to a CSV file
df = pd.DataFrame(matrix, columns=[f"Month_{i:02}" for i in range(1, 13)])
df.index = [f"File_{i:02}" for i in range(1, rows + 1)]
output_file = "monthly_durations.csv"
df.to_csv(output_file)

print(f"Matrix saved to {output_file}.")
