import csv

# Input and output filenames
input_filename = 'output.csv'
output_filename = '23-06-24_out.csv'

# Open the input file and output file
with open(input_filename, mode='r') as infile, open(output_filename, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    
    # Get column names from the CSV file
    fieldnames = ['id', 'vertex', 'latitude', 'longitude', 'Al','Si','Mg','Fe','Ti','Ca','Na','O']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the header to the output file
    writer.writeheader()

    # Iterate through each row in the input file
    for row in reader:
        id_value = row['id']
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
                'id': id_value,
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

print("CSV conversion completed!")

