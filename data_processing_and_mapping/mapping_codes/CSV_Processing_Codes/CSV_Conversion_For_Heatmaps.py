import csv

# Input and output filenames
input_filename = 'out_new.csv'
output_filename = 'out_new_heat.csv'

# Open the input file and output file
with open(input_filename, mode='r') as infile, open(output_filename, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    
    # Get column names from the CSV file
    fieldnames = ['id', 'latitude', 'longitude', 'Al','Si','Mg','Fe','Ti','Ca','Na','O']
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
        latsum=0
        lonsum=0
        for i in range(4):

            latsum = latsum+float(row[f'V{i}_lat'])
            lonsum = lonsum+float(row[f'V{i}_lon'])
            
            # Write the new row to the output file
        writer.writerow({
            'id': id_value,
            'latitude': latsum/4,
            'longitude': lonsum/4,
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

