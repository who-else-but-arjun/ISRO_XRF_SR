import os

def process_fits_files(input_directory, output_file):
    # Open the output file to write the extracted times
    with open(output_file, 'w') as outfile:
        # Iterate through all files in the directory
        for filename in sorted(os.listdir(input_directory)):
            # Check if the file is a FITS file
            if filename.endswith('.fits'):
                try:
                    # Extract the times from the filename
                    parts = filename.split('_')
                    time_range = parts[-1].split('.')[0]
                    start_time, end_time = time_range.split('-')

                    # Format times for output
                    start_time_formatted = (
                        f"{start_time[:4]}-{start_time[4:6]}-{start_time[6:8]}T"
                        f"{start_time[9:11]}:{start_time[11:13]}:{start_time[13:15]}.{start_time[15:]}"
                    )
                    end_time_formatted = (
                        f"{end_time[:4]}-{end_time[4:6]}-{end_time[6:8]}T"
                        f"{end_time[9:11]}:{end_time[11:13]}:{end_time[13:15]}.{end_time[15:]}"
                    )

                    # Write the formatted times to the output file
                    outfile.write(f"{start_time_formatted} {end_time_formatted}\n")
                
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

# Define input directory and output file
input_directory = 'combined/L1_ADDED_FILES_TIME/'  # Replace with your actual directory
output_file = 'eminem.txt'

# Process the FITS files
process_fits_files(input_directory, output_file)

print(f"Times have been written to {output_file}")
