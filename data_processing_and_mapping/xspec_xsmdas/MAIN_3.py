from xspec import *
import xraylib
import os
import numpy as np
from datetime import datetime
from astropy.io import fits
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V2 import xrf_comp
import geopandas as gpd
from shapely.geometry import Polygon

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

# Read the abundances from the file and store them in an array
abundances = []
with open('abundances.txt', 'r') as file:
    for line in file:
        # Convert each line to a float and append to the array
        abundances.append(float(line.strip()))  # strip() removes any extra whitespace/newline

# Print the read abundances
print("Abundances read from file:", abundances)

# atomic numbers etc
element_name=["Fe","Ti","Ca","Si","Al","Mg","Na","O"]
atomic_number=[26,22,20,14,13,12,11,8]
atomic_weight=[55.847,47.9,40.08,28.086,26.982,24.305,22.989,15.9994]
k_alpha=[6.403,4.510,3.691,1.739,1.486,1.253,1.040,0.525]
k_beta=[7.112,4.966,4.038,1.839,1.559,1.303,1.070,0.543]

# import spectrum value arrays (eg. energy_array_spectrum)

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

from get_xrf_lines_V1 import get_xrf_lines
xrf_lines = get_xrf_lines(
    at_no=atomic_number,  # Fe, Ti, Ca, Si, Al, Mg, Na, O
    k_shell=xraylib.K_SHELL,
    k_lines=[xraylib.KL1_LINE, xraylib.KL2_LINE],
    l1_shell=xraylib.L1_SHELL,
    l1_lines=[xraylib.L1L2_LINE, xraylib.L1L3_LINE],
    l2_shell=xraylib.L2_SHELL,
    l2_lines=[xraylib.L2L3_LINE, xraylib.L2M1_LINE],
    l3_shell=xraylib.L3_SHELL,
    l3_lines=[xraylib.L3M1_LINE, xraylib.L3M2_LINE]
)

# energy_low = energy_array_scatter_low
# energy_high = energy_array_scatter_high
# add a check somehow
solar_energy = energy_array_spectrum
solar_counts = counts_array_spectrum
#scatter calculation function using 


from get_constants_xrf_new_V2 import get_constants_xrf
const_xrf = get_constants_xrf(
    energy=solar_energy,  # Energy array from the solar spectrum file
    at_no=atomic_number,
    weight=abundances,  # Input elemental weight fractions
    xrf_lines=xrf_lines
)

static_parameter_file = "static_par_localmodel.txt"
solar_ang = None
emis_angle = None

with open(static_parameter_file, "r") as file:
    lines = file.readlines()
    if len(lines) >= 3:
        solar_ang = float(lines[1].strip())  
        emis_angle = float(lines[2].strip()) 

if solar_ang is not None and emis_angle is not None:
    i_a = 90 - solar_ang
    e_a = 90 - emis_angle

from xrf_comp_new_V2 import xrf_comp
xrf_struc = xrf_comp(
    energy=solar_energy, counts=solar_counts, i_angle=i_a,
    e_angle=e_a, at_no=[26, 22, 20, 14, 13, 12, 11, 8],
    weight=abundances, xrf_lines=xrf_lines, const_xrf=const_xrf
)


fid = open(static_parameter_file,"r")
finfo_full = fid.read()
finfo_split = finfo_full.split('\n')
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])

from define_xrf_localmodel import xrf_localmodel
import xspec
xrf_localmodel_ParInfo = (f"Wt_Fe \"\" 5 1 1 20 20 1e-2",f"Wt_Ti \"\" 1 1e-6 1e-6 20 20 1e-2",f"Wt_Ca \"\" 9 5 5 20 20 1e-2",f"Wt_Si \"\" 21 15 15 35 35 1e-2",f"Wt_Al \"\" 14 5 5 20 20 1e-2",f"Wt_Mg \"\" 5 1e-6 1e-6 20 20 1e-2",f"Wt_Na \"\" 0.5 1e-6 1e-6 5 5 1e-2",f"Wt_O \"\" 45 30 30 60 60 1e-2")
xspec.AllModels.addPyMod(xrf_localmodel, xrf_localmodel_ParInfo, 'add')

import xspec
spec=xspec.Spectrum(file_path,backFile='background_allevents.fits',respFile='class_rmf_v1.rmf',arfFile='class_arf_v1.arf')
xspec.AllData.ignore("1-68, 312-2048")
full_model = "atable{tbmodel.fits}+xrf_localmodel"
mo = xspec.Model(full_model)
mo(9).values = "45.0"
mo(9).frozen = True
mo(5).link = '100 - (2+3+4+6+7+8+9)'

optimization_filepath='optimization.txt'

Fit.nIterations = 3
Fit.perform()

# Plot.device="/xs"  # Set the plotting device to an X11 window.
# Plot("ldata")      # Plot the data with the fitted model overlaid.

chi_square = Fit.statistic
degrees_of_freedom = Fit.dof    
reduced_chi_square = chi_square / degrees_of_freedom

delete_file1 = (
    f"ch2_xsm_"
    f"{start_time.year:04d}{start_time.month:02d}{start_time.day:02d}_"
    f"{start_time.hour:02d}{start_time.minute:02d}{start_time.second:02d}-"
    f"{stop_time.hour:02d}{stop_time.minute:02d}{stop_time.second:02d}.arf"
)

delete_file2 = (
    f"ch2_xsm_"
    f"{start_time.year:04d}{start_time.month:02d}{start_time.day:02d}_"
    f"{start_time.hour:02d}{start_time.minute:02d}{start_time.second:02d}-"
    f"{stop_time.hour:02d}{stop_time.minute:02d}{stop_time.second:02d}.pha"
)

if os.path.exists(delete_file1):
  os.remove(delete_file1)
  print("deleted file 1")
else:
  print("The file does not exist")

if os.path.exists(delete_file2):
  os.remove(delete_file2) 
  print("deleted file 2")
else:
  print("The file does not exist")


if(reduced_chi_square<10):
    Fit.nIterations = 7
    Fit.perform()
    chi_square = Fit.statistic
    degrees_of_freedom = Fit.dof    
    reduced_chi_square = chi_square / degrees_of_freedom
    with open(optimization_filepath, "w") as f:
        f.write("1")
        print("Data written to .txt")
else:
    # MODELOP in the making
    with open(optimization_filepath, "w") as f:
        f.write("0")
        print("Data written to .txt")

print(f"Reduced Chi-Square: {reduced_chi_square}")
chi_squared_path="chi_squared.txt"

with open(chi_squared_path,'w') as file:
    file.write(f"{reduced_chi_square:.6f}")

abundances=[]
abundances.append(mo(2).values)  # Wt_Fe
abundances.append(mo(3).values)  # Wt_Ti
abundances.append(mo(4).values)  # Wt_Ca
abundances.append(mo(5).values)  # Wt_Si
abundances.append(mo(6).values)  # Wt_Al
abundances.append(mo(7).values)  # Wt_Mg
abundances.append(mo(8).values)  # Wt_Na
abundances.append(mo(9).values)  # Wt_O
abundances = np.array(abundances)

# writing the found abundances in the common text file called abundances.txt

# Write the abundances array to a file
with open('abundances.txt', 'w') as file:
    for i in abundances:
        file.write(f"{i[0]}\n")

print("Abundances have been written to abundances.txt")

delete_file1 = (
    f"ch2_xsm_"
    f"{start_time.year:04d}{start_time.month:02d}{start_time.day:02d}_"
    f"{start_time.hour:02d}{start_time.minute:02d}{start_time.second:02d}-"
    f"{stop_time.hour:02d}{stop_time.minute:02d}{stop_time.second:02d}.fits"
) 


