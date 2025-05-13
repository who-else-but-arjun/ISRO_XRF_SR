from xspec import *
import xraylib
import os
import numpy as np
from datetime import datetime
from scipy import interpolate
from astropy.io import fits
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V2 import xrf_comp
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# prerequisites from previous file

element_name=["Fe","Ti","Ca","Si","Al","Mg","Na","O"]
atomic_number=[26,22,20,14,13,12,11,8]
atomic_weight=[55.847,47.9,40.08,28.086,26.982,24.305,22.989,15.9994]
k_alpha=[6.403,4.510,3.691,1.739,1.486,1.253,1.040,0.525]
k_beta=[7.112,4.966,4.038,1.839,1.559,1.303,1.070,0.543]

def extract_times_and_calculate_tstart_tstop(file_path, tref):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    with open(file_path, "r") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError("time.txt is empty or improperly formatted.")
    start_time_str, stop_time_str = line.split()
    try:
        start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
    try:
        stop_time = datetime.strptime(stop_time_str, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        stop_time = datetime.strptime(stop_time_str, "%Y-%m-%dT%H:%M:%S")
    return start_time, stop_time

file_path_time = "time.txt"  
tref = datetime(2017, 1, 1)
start_time, stop_time = extract_times_and_calculate_tstart_tstop(file_path_time, tref)

file_path = (
    f"combined/L1_ADDED_FILES_TIME/ch2_cla_L1_time_added_"
    f"{start_time.year:04d}{start_time.month:02d}{start_time.day:02d}T"
    f"{start_time.hour:02d}{start_time.minute:02d}{start_time.second:02d}"
    f"{start_time.microsecond // 1000:03d}-"
    f"{stop_time.year:04d}{stop_time.month:02d}{stop_time.day:02d}T"
    f"{stop_time.hour:02d}{stop_time.minute:02d}{stop_time.second:02d}"
    f"{stop_time.microsecond // 1000:03d}.fits"
)  


solar_angle=0
emis_angle=0
with fits.open(file_path) as hdul:
    header = hdul[1].header  # Accessing the header of the second HDU (HDU 1)
    # Extracting specific values
    solar_angle = header.get('SOLARANG', 'N/A')
    emis_angle = header.get('EMISNANG', 'N/A')

# calculated using qdp
energy_array_spectrum=[]
counts_array_spectrum=[]
error_array_spectrum=[]

# Open the file and read line by line
# dont forget to change file path
with open('spectrum.qdp', 'r') as file:
    for line in file:
        # Split the line into columns
        columns = line.split()
        
        # Skip lines that don't have enough columns
        if len(columns) < 5:
            continue
        
        # Check if the 5th column is "NO"
        if columns[4] == "NO":
            break  # Stop reading when "NO" is encountered

        # Append the values to respective lists
        energy_array_spectrum.append(float(columns[0]))  # 1st column
        counts_array_spectrum.append(float(columns[4]))  # 5th column
        error_array_spectrum.append(float(columns[1]))  # 2nd column

    # just making np arrays
    energy_array_spectrum=np.array(energy_array_spectrum)
    counts_array_spectrum=np.array(counts_array_spectrum)
    error_array_spectrum=np.array(error_array_spectrum)

fullpath = os.path.abspath(__file__)
script_path, filename = os.path.split(fullpath)

def sigmabyrho_of_element(at_no):
    elename_string=""
    for i in range(len(atomic_number)):
        if(atomic_number[i]==at_no):
            elename_string=element_name[i]
            break
    filename = os.path.join(script_path, f'data_constants/ffast/ffast_{int(at_no):d}_{elename_string.lower()}.txt')

    # Arrays to store extracted values
    energy = []
    sigma_by_rho = []

    # Open and process the file
    with open(filename, 'r') as file:
        is_data_section = False  # Flag to identify the data section
        for line in file:
            # Check if the line starts the data section
            if "Form Factors, Attenuation and Scattering Cross-sections" in line:
                is_data_section = True  # Start reading the data
                continue
            
            # Skip until the actual data rows begin
            if is_data_section and line.strip().startswith('keV'):
                continue  # Skip the header line
                
            if is_data_section and line.strip():  # Process only non-empty lines
                try:
                    # Extract the relevant columns (5th and 6th)
                    columns = line.split()
                    energy.append(float(columns[0]))
                    sigma_by_rho.append(float(columns[4]))  # Column 5
                except (IndexError, ValueError):
                    # Skip lines that do not contain valid data
                    continue

    # Convert lists to numpy arrays
    energy = np.array(energy)
    sigma_by_rho = np.array(sigma_by_rho)
    
    return energy,sigma_by_rho

def mubyrho_of_element(at_no):
    elename_string=""
    for i in range(len(atomic_number)):
        if(atomic_number[i]==at_no):
            elename_string=element_name[i]
            break
    filename = os.path.join(script_path, f'data_constants/ffast/ffast_{int(at_no):d}_{elename_string.lower()}.txt')

    # Arrays to store extracted values
    energy = []
    mu_by_rho = []

    # Open and process the file
    with open(filename, 'r') as file:
        is_data_section = False  # Flag to identify the data section
        for line in file:
            # Check if the line starts the data section
            if "Form Factors, Attenuation and Scattering Cross-sections" in line:
                is_data_section = True  # Start reading the data
                continue
            
            # Skip until the actual data rows begin
            if is_data_section and line.strip().startswith('keV'):
                continue  # Skip the header line
                
            if is_data_section and line.strip():  # Process only non-empty lines
                try:
                    # Extract the relevant columns (5th and 6th)
                    columns = line.split()
                    energy.append(float(columns[0]))
                    mu_by_rho.append(float(columns[5]))  # Column 6
                except (IndexError, ValueError):
                    # Skip lines that do not contain valid data
                    continue

    # Convert lists to numpy arrays
    energy = np.array(energy)
    mu_by_rho = np.array(mu_by_rho)

    return energy,mu_by_rho

# Returns an interpolation function that can give sigma by rho value at any energy
def sigmabyrho_interpolated(at_no,x):
    energy_arr,sigma_by_rho=sigmabyrho_of_element(at_no)
    interpolation_sigma_by_rho=interpolate.interp1d(energy_arr, sigma_by_rho)(x)
    return interpolation_sigma_by_rho

# returns an interpolation function that can give mu by rho value at any energy
def mubyrho_interpolated(at_no,x):
    energy_arr,mu_by_rho=sigmabyrho_of_element(at_no)
    interpolation_mu_by_rho=interpolate.interp1d(energy_arr, mu_by_rho)(x)
    return interpolation_mu_by_rho

def spectrum_interpolated(energy_spectrum_qdp, counts_spectrum_qdp,x):
    spectrum_interpolation_function=interpolate.interp1d(energy_spectrum_qdp, counts_spectrum_qdp)(x)
    return spectrum_interpolation_function

def cosec(x):
    return 1/np.sin(x*((np.pi)/180))

def solar_to_scatter_continuum(abundances):
    no_elements=len(atomic_number) # 8

    # energy array for final needs
    energy_array = np.arange(1.25, 7.15, 0.01)  # Use 7.21 to include 7.2 in the array

    interpolated_counts=[]
    # print(spectrum_interpolated(energy_array_spectrum,counts_array_spectrum,energy_array))
    # exit()

    for i in range(len(energy_array)):
        interpolated_counts.append(spectrum_interpolated(energy_array_spectrum,counts_array_spectrum,energy_array[i]))

    #   just for checking
    # plt.plot(energy_array,interpolated_counts)
    # plt.title("INTERPOLATED")
    # plt.yscale('log')
    # plt.show()


    # finally calculating the scatter continuum
    Scatter_continuum=[]
    for i in range(len(energy_array)):
        # weighted averaging
        sigma_by_rho_total=0
        mu_by_rho_total=0
        for ele in range(no_elements):
            sigmabyrho_interpolation_function_element=sigmabyrho_interpolated(atomic_number[ele],energy_array[i])
            mubyrho_interpolation_function_element=mubyrho_interpolated(atomic_number[ele],energy_array[i])
            sigma_by_rho_total+=(abundances[ele]/100)*sigmabyrho_interpolation_function_element
            mu_by_rho_total+=(abundances[ele]/100)*mubyrho_interpolation_function_element
        
        numerator_scattercontinuum=(interpolated_counts[i]*sigma_by_rho_total)
        denominator_scattercontinuum=(mu_by_rho_total*(cosec(solar_angle)+cosec(emis_angle)))
        Scatter_continuum.append(numerator_scattercontinuum/denominator_scattercontinuum)

    return energy_array, Scatter_continuum

# SCATTER CONTINUUM
# Read the abundances from the file and store them in an array
abundances = []
with open('abundances.txt', 'r') as file:
    for line in file:
        # Convert each line to a float and append to the array
        abundances.append(float(line.strip()))  # strip() removes any extra whitespace/newline

# Print the read abundances
print("Abundances read from file:", abundances)

energy_array_scatter,scatter_continuum_counts= solar_to_scatter_continuum(abundances)
bin_size=0.01

# plot for check
# plt.plot(energy_array_scatter,scatter_continuum_counts)
# plt.title("SCATTER CONTINUUM")
# plt.yscale('log')
# plt.show()

bin_array=np.empty(len(energy_array_scatter))
bin_array.fill(bin_size/2)

energy_array_scatter_low = np.round(energy_array_scatter-bin_array,decimals=3)
energy_array_scatter_high = np.round(energy_array_scatter+bin_array,decimals=3)

filename="ftools_infile_scatter.txt"

with open(filename, "w") as f:
    # Write each row of the arrays into the file
    for a1, a2, a3 in zip(energy_array_scatter_low, energy_array_scatter_high, scatter_continuum_counts):
        f.write(f"{a1}\t{a2}\t{a3}\n")

print("Data written to ftools_infile_scatter.txt")









