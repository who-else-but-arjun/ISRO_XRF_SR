import os
from datetime import datetime
coordinates_file = "coordinates.txt"
abundances_file = "abundances.txt"
chi_squared_file = "chi_squared.txt"
csv_file = "final_abundances.csv"

# Read coordinates (8 values, one per line)
with open(coordinates_file, "r") as coords:
    coordinates = [line.strip() for line in coords if line.strip()]
    if len(coordinates) != 8:
        raise ValueError(f"Expected 8 values in {coordinates_file}, but got {len(coordinates)}")

# Read the abundances file
with open(abundances_file, "r") as abundances:
    # Extract non-empty, stripped lines
    abundances_values = [float(line.strip()) for line in abundances if line.strip()]
    
    # Ensure there are exactly 8 values
    if len(abundances_values) != 8:
        raise ValueError(f"Expected 8 values in {abundances_file}, but got {len(abundances_values)}")
    
    # Separate the first 7 values and the last value
    first_seven = abundances_values[:7]
    last_value = abundances_values[7]
    
    # Combine normalized values with the last value
    abundances_values = first_seven + [last_value]

# Output the resulting values
print("Normalized abundances:", abundances_values)

# Read chi-squared value (single float value, first line)
with open(chi_squared_file, "r") as chi_file:
    chi_squared_value = chi_file.readline().strip()
    try:
        chi_squared_value = float(chi_squared_value)
    except ValueError:
        raise ValueError(f"Invalid float value in {chi_squared_file}: {chi_squared_value}")

# Combine values
row_values = coordinates + abundances_values + [f"{chi_squared_value:.6f}"]

print(row_values)

# Append to CSV without leaving a blank line
if(chi_squared_value<5):
    print("hello")
    with open(csv_file, "a", newline="") as csv:
        csv.write(",".join(map(str, row_values)) + "\n")




