import pandas as pd
import os

# Define the new column names as per your updated structure (17 columns)
columns = ['id', 'V0_LAT', 'V0_LON', 'V1_LAT', 'V1_LON', 'V2_LAT', 'V2_LON', 'V3_LAT', 'V3_LON', 'Al', 'Si', 'Mg', 'Fe', 'Ti', 'Ca', 'Na', 'O', 'Chi_sq']

# Define the folder containing the CSV files
folder_path = 'Data Analysis'  # Replace with the path to your folder

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Initialize the starting ID for continuity
current_max_id = 0

# Iterate over each CSV file and process them
for file in csv_files:
    # Load the current CSV file without headers
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, header=None)
    
    # Check if the number of columns matches the expected number (17 columns now)
    if len(df.columns) != len(columns) - 1:  # -1 because 'id' is being added
        print(f"Warning: File '{file}' has {len(df.columns)} columns, expected {len(columns) - 1}. Skipping this file.")
        continue  # Skip files with unexpected column numbers
    
    # Insert the 'id' column at the leftmost position
    df.insert(0, 'id', range(current_max_id + 1, current_max_id + 1 + len(df)))  # Ensure continuous ids
    
    # Rename the columns
    df.columns = columns
    
    # Save the corrected CSV file (converted file)
    converted_file_path = os.path.join(folder_path, f"converted_{file}")
    df.to_csv(converted_file_path, index=False)
    print(f"Converted file saved as: {converted_file_path}")
    
    # Update the current_max_id for the next file
    current_max_id = df['id'].max()
    
    # Append the data to the combined DataFrame
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(os.path.join(folder_path, 'combined_file.csv'), index=False)

print("All CSV files have been successfully combined into 'combined_file.csv'")
