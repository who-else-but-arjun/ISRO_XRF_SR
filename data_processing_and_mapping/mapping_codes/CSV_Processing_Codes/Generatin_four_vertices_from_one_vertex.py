import pandas as pd

def calculate_vertices(lat_center, lon_center, half_side_length):
    """
    Calculate the vertices of a square given the center coordinates and half the side length.
    """
    lat0, lon0 = lat_center + half_side_length, lon_center - half_side_length
    lat1, lon1 = lat_center + half_side_length, lon_center + half_side_length
    lat2, lon2 = lat_center - half_side_length, lon_center + half_side_length
    lat3, lon3 = lat_center - half_side_length, lon_center - half_side_length
    return lat0, lon0, lat1, lon1, lat2, lon2, lat3, lon3

# Read input CSV
input_csv_path = "subregion_0_2.csv"  # Update with your file path
output_csv_path = "output.csv"  # File path for the transformed data
side_length = 0.06596  # Example side length, adjust as needed
half_side_length = side_length / 2

# Load the CSV
df = pd.read_csv(input_csv_path)

# Add the new columns for vertices
df[["lat0", "lon0", "lat1", "lon1", "lat2", "lon2", "lat3", "lon3"]] = df.apply(
    lambda row: calculate_vertices(row['lat_center'], row['lon_center'], half_side_length),
    axis=1,
    result_type='expand'
)

# Save the transformed data to a new CSV
df.to_csv(output_csv_path, index=False)
print(f"Transformed data saved to {output_csv_path}")

import csv

# Input and output filenames
input_filename = f'output.csv'  # Replace with your file name
output_filename = 'output_with_vertices.csv'

# Open the input file and output file
with open(input_filename, mode='r') as infile, open(output_filename, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    
    # Define column headers for the output
    fieldnames = ['id', 'vertex', 'latitude', 'longitude','Al','Si','Mg','Fe','Ti','Ca','Na','O']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the header to the output file
    writer.writeheader()

    # Initialize ID counter
    id_counter = 1

    # Iterate through each row in the input file
    for row in reader:
        mg = row['Mg']
        al = row['Al']
        si = row['Si']
        fe = row['Fe']
        ti = row['Ti']
        na = row['Na']
        o = row['O']
        ca = row['Ca']
        # Create a new row for each vertex (V0, V1, V2, V3)
        for i in range(4):
            vertex = f'V{i}'
            latitude = row[f'lat{i}']
            longitude = row[f'lon{i}']
            
            # Write the new row to the output file
            writer.writerow({
                'id': id_counter,
                'vertex': vertex,
                'latitude': latitude,
                'longitude': longitude,
                'Mg': mg,
                'Al': al,
                'Si': si,
                'Fe': fe,
                'Ca': ca,
                'Ti': ti,
                'O': o,
                'Na': na,
            })
        
        # Increment the ID counter
        id_counter += 1

print("CSV conversion completed!")

