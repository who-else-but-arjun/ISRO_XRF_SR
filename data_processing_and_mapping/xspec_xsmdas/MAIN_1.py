from xspec import *
import xraylib
import os
import numpy as np
from datetime import datetime
import corner
import matplotlib.pyplot as plt
from scipy.interpolate import barycentric_interpolate
from astropy.io import fits
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V2 import xrf_comp
import geopandas as gpd
from shapely.geometry import Polygon

# Code for xsmdsas
# ----------------------------------------------------------------------------


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
tstart=(start_time-tref).total_seconds()
tstop=(stop_time-tref).total_seconds()

base1=f"combined/{start_time.day:02d}"

const1 = f"/xsm/data/{start_time.year:02d}/{start_time.month:02d}/{start_time.day:02d}"
const1 = base1+const1
l1dir = f"{const1}/raw"
l2dir = f"{const1}/calibrated"

print("L1 Directory:", l1dir)
print("L2 Directory:", l2dir)

base=f"ch2_xsm_{start_time.year:02d}{start_time.month:02d}{start_time.day:02d}_v1"

l1file=l1dir+'/'+base+'_level1.fits'
hkfile=l1dir+'/'+base+'_level1.hk'
safile=l1dir+'/'+base+'_level1.sa'
gtifile=l2dir+'/'+base+'_level2.gti'

# Have to make name of this file using date and time under consideration
specbase=f"ch2_xsm_{start_time.year:02d}{start_time.month:02d}{start_time.day:02d}_{start_time.hour:02d}{start_time.minute:02d}{start_time.second:02d}-{stop_time.hour:02d}{stop_time.minute:02d}{stop_time.second:02d}"
specfile=specbase+'.pha'

genspec_command="xsmgenspec l1file="+l1file+" specfile="+specfile+" spectype='time-integrated'"+ \
" tstart="+str(tstart)+" tstop="+str(tstop)+" hkfile="+hkfile+" safile="+safile+" gtifile="+gtifile

s=os.system(genspec_command)

# Clear any previously loaded data and models

AllData.clear()
AllModels.clear()

# Load spectrum (ARF, RMF, background etc will be loaded automatically if specified in spectrum header)
import xspec
spec = xspec.Spectrum(specfile)


# Set plot device, x-axis to energy instead of channel, plot spectrum
# Plot.device = '/xw'    
# Plot.xAxis = 'keV'

# Plot('ld')

spec.ignore("**-1.2 7.2-**")
# Plot('ld')

# define the model
m1 = Model("vapec+powerlaw")
m1.setPars(6.5, 9.77E-02, 3.98E-04, 1.00E-04, 8.51E-04, 1.29E-04, 1.29E-04, 3.80E-05, 2.95E-06, 3.55E-05, 1.62E-05, 4.47E-06, 2.29E-06, 3.24E-05, 1.78E-06)
m1.vapec.He.frozen=False
m1.vapec.C.frozen=False
m1.vapec.N.frozen=False
m1.vapec.O.frozen=False
m1.vapec.Ne.frozen=False
m1.vapec.Mg.frozen=False
m1.vapec.Al.frozen=False
m1.vapec.Si.frozen=False
m1.vapec.S.frozen=False
m1.vapec.Ar.frozen=False
m1.vapec.Ca.frozen=False
m1.vapec.Fe.frozen=False
m1.vapec.Ni.frozen=False

# # do the fit
# Fit.perform()

# # plot data, model and del-chi
# # Plot('ld','delc')

# # Free some parameters that are frozen (Mg, Si, and S)
m1.show()

# m1.vvapec.Mg.frozen=False
# m1.vvapec.Si.frozen=False
# m1.vvapec.S.frozen=False

## Note: Abundances of other elements also should be set to coronal values as required. 
##       Default abundances in apec are not coronal

# Fit spectrum again
Fit.perform()

# xspec.Plot.device = "/xs"  # Set plotting device ("/xs" for screen, or "none")
# xspec.Plot.xLog = False    # Example of plot customization
# xspec.Plot("ldata")        # Plot observed data

# Save to QDP file
# xspec.Plot.addCommand("wdata spectrum.qdp")  # Writes data to a file

xspec.Plot.device="/null" 
# xspec.Plot.xLog= False
xspec.Plot.xAxis="keV"
xspec.Plot("ldata")

print("running spectrum.qdp making")
# xspec.Plot.addCommand("WData spectrum.qdp")
x_vals=xspec.Plot.x()
x_errs=xspec.Plot.xErr()
y_vals=xspec.Plot.y()
y_errs=xspec.Plot.yErr()
model_vals= xspec.Plot.model()

#Write the data to spectrum.qdp
with open("spectrum.qdp","w") as f:
    for i in range(len(x_vals)):
        f.write(f"{x_vals[i]} {x_errs[i]} {y_vals[i]} {y_errs[i]} {model_vals[i]}\n")

print("running spectrum.qdp making")
# Plot('ld','delc')

# Show best-fit  parameters
# m1.show()
if os.path.exists("model.xcm"):
    os.remove("model.xcm")
# Load the data and create a model
# Save the model to an XCM file
xspec.Xset.save("model.xcm", info="m")  # Save in .xcm format

# -----------------------------------------------------------------------------

# Fe Ti Ca Si Al Mg Na O 
# initializing weights of a region based on highland and mare
init_weights_highland=np.array([2.58, 0.66, 13.15, 22.41, 13.65, 0.18, 1.37, 45])
init_weights_mare=np.array([9.72, 4.37, 7.00, 18.76, 7.40, 7.23, 1.37, 45])
# Filepath to the FITS file

# combined fits files path
file_path = (
    f"combined/L1_ADDED_FILES_TIME/ch2_cla_L1_time_added_"
    f"{start_time.year:04d}{start_time.month:02d}{start_time.day:02d}T"
    f"{start_time.hour:02d}{start_time.minute:02d}{start_time.second:02d}"
    f"{start_time.microsecond // 1000:03d}-"
    f"{stop_time.year:04d}{stop_time.month:02d}{stop_time.day:02d}T"
    f"{stop_time.hour:02d}{stop_time.minute:02d}{stop_time.second:02d}"
    f"{stop_time.microsecond // 1000:03d}.fits"
)

# just appending the just completed files name to done_filenames.txt
done_files="done_filenames.txt"

with open(done_files, "a") as done_file:
    done_file.write(file_path+'\n')
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

    # plotting for check
    # plt.plot(energy_array_spectrum,counts_array_spectrum)
    # plt.yscale('log')
    # plt.show()

# MODELOP in the making
with open("modelop.txt", "w") as f:
    # Write each row of the arrays into the file
    for a1, a2, a3 in zip(energy_array_spectrum, error_array_spectrum, counts_array_spectrum):
        f.write(f"{a1}\t{a2}\t{a3}\n")

print("Data written to modelop.txt")

# using this function to calculate the percentage of highland and mare in a particular region
# we will be using this to calculate the weighted initialized weights
def analyze_overlap_area(v0_lat, v0_lon, v1_lat, v1_lon, 
                         v2_lat, v2_lon, v3_lat, v3_lon):
    """
    Analyze the area overlap between the quadrilateral region and maria polygons.
    """
    # Load the shapefile containing maria regions
    shapefile_path = 'LROC_GLOBAL_MARE_180.SHP'
    gdf = gpd.read_file(shapefile_path)
    
    # Check and fix invalid geometries in the shapefile
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    
    # Create the quadrilateral polygon
    quadrilateral_coords = [
        (v0_lon, v0_lat), 
        (v1_lon, v1_lat), 
        (v2_lon, v2_lat), 
        (v3_lon, v3_lat)
    ]
    quadrilateral = Polygon(quadrilateral_coords)
    
    # Check if the quadrilateral is valid, and fix it if necessary
    if not quadrilateral.is_valid:
        quadrilateral = quadrilateral.buffer(0)
    
    # Total area of the quadrilateral
    quadrilateral_area = quadrilateral.area
    
    # Calculate maria overlap area
    maria_overlap_area = sum(gdf['geometry'].intersection(quadrilateral).area)
    
    # Calculate highland area (remaining area in the quadrilateral)
    highland_area = quadrilateral_area - maria_overlap_area
    
    # Calculate percentages
    maria_percentage = (maria_overlap_area / quadrilateral_area) * 100 if quadrilateral_area > 0 else 0
    highland_percentage = (highland_area / quadrilateral_area) * 100 if quadrilateral_area > 0 else 0
    
    # Return results
    return {
        'maria_percentage': maria_percentage,
        'highland_percentage': highland_percentage
    }  


def find_weighted_init(v0_lat,v1_lat,v2_lat,v3_lat,v0_lon,v1_lon,v2_lon,v3_lon):
    temp = analyze_overlap_area(v0_lat,v0_lon,v1_lat,v1_lon,v2_lat,v2_lon,v3_lat,v3_lon)
    maria_fraction, highland_fraction = temp['maria_percentage'], temp['highland_percentage']
    print(f"Maria % = {maria_fraction}")
    print(f"Highland % = {highland_fraction}")

    weighted_init_abundances=(maria_fraction*init_weights_mare+highland_fraction*init_weights_highland)/100
    print("these are weighted abundance initial",weighted_init_abundances)
    return weighted_init_abundances



# Open the FITS file and extract the required header values
solar_angle=0
emis_angle=0

v0_lat=0
v1_lat=0
v2_lat=0
v3_lat=0

v0_lon=0
v1_lon=0
v2_lon=0
v3_lon=0

altitude=0
exposure=0

with fits.open(file_path) as hdul:
    header = hdul[1].header  # Accessing the header of the second HDU (HDU 1)

    # Extracting specific values
    solar_angle = header.get('SOLARANG', 'N/A')
    emis_angle = header.get('EMISNANG', 'N/A')

    # Extracting corner coordinates
    v0_lat = header.get('V0_LAT', 'N/A')
    v1_lat = header.get('V1_LAT', 'N/A')
    v2_lat = header.get('V2_LAT', 'N/A')
    v3_lat = header.get('V3_LAT', 'N/A')

    v0_lon = header.get('V0_LON', 'N/A')
    v1_lon = header.get('V1_LON', 'N/A')
    v2_lon = header.get('V2_LON', 'N/A')
    v3_lon = header.get('V3_LON', 'N/A')

    altitude=header.get("SAT_ALT",'N/A')
    exposure=header.get("EXPOSURE",'N/A')
    # just checking
    print("values picked: ")
    print(solar_angle)
    print("coordinates: ")
    print(v0_lat,v0_lon)
    print(v1_lat,v1_lon)
    print(v2_lat,v2_lon)
    print(v3_lat,v3_lon)


#CHECK FOR 96 SECONDS EXPOSURE TIME AND HENCE A CHECK IF 12 FILES WERE COMBINED OR NOT
optimization_filepath='optimization.txt'
if(exposure==96):
    with open(optimization_filepath, "w") as f:
        f.write("1")
        print("Data written to .txt")
else:
    # MODELOP in the making
    with open(optimization_filepath, "w") as f:
        f.write("0")
        print("Data written to .txt")


# just making the coordinates file coordinates.txt
coordinates_file = "coordinates.txt"

with open(coordinates_file, "w") as coords:
    coords.write(f"{v0_lat}\n")
    coords.write(f"{v0_lon}\n")
    coords.write(f"{v1_lat}\n")
    coords.write(f"{v1_lon}\n")
    coords.write(f"{v2_lat}\n")
    coords.write(f"{v2_lon}\n")
    coords.write(f"{v3_lat}\n")
    coords.write(f"{v3_lon}\n")

element_name=["Fe","Ti","Ca","Si","Al","Mg","Na","O"]
atomic_number=[26,22,20,14,13,12,11,8]
atomic_weight=[55.847,47.9,40.08,28.086,26.982,24.305,22.989,15.9994]
k_alpha=[6.403,4.510,3.691,1.739,1.486,1.253,1.040,0.525]
k_beta=[7.112,4.966,4.038,1.839,1.559,1.303,1.070,0.543]

fullpath = os.path.abspath(__file__)
script_path, filename = os.path.split(fullpath)

# Writing relevant data into the static_par_localmodel file
# Define the 4 static values
static_values = [solar_angle, emis_angle, altitude, exposure]

# Write the static values into the file
with open('static_par_localmodel.txt', 'w') as file:
    file.write("modelop.txt\n")
    for value in static_values:
        file.write(f"{value}\n")

print("Static values have been written to static_par_localmodel.txt")


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



# returns an interpolation function that can give sigma by rho value at any energy
def sigmabyrho_interpolated(at_no):
    energy_arr,sigma_by_rho=sigmabyrho_of_element(at_no)
    interpolation_sigma_by_rho=barycentric_interpolate(energy_arr, sigma_by_rho)
    return interpolation_sigma_by_rho

# returns an interpolation function that can give mu by rho value at any energy
def mubyrho_interpolated(at_no):
    energy_arr,mu_by_rho=sigmabyrho_of_element(at_no)
    interpolation_mu_by_rho=barycentric_interpolate(energy_arr, mu_by_rho)
    return interpolation_mu_by_rho

def spectrum_interpolated(energy_spectrum_qdp, counts_spectrum_qdp):
    spectrum_interpolation_function=barycentric_interpolate(energy_spectrum_qdp, counts_spectrum_qdp)
    return spectrum_interpolation_function

def cosec(x):
    return 1/np.sin(x)

# Calculating scatter continuum total
# Interpolation functions for sigma by rho and mu by rho are passed
# Terrane will be highland or mare
# Returns an energy_arr and the scattered continuum corresponding to the energy values

def solar_to_scatter_continuum(abundances):
    no_elements=len(atomic_number) # 8

    # energy array for final needs
    energy_array = np.arange(1.2, 7.21, 0.01)  # Use 7.21 to include 7.2 in the array

    # interpolated_counts calculation
    interpolated_counts_function=spectrum_interpolated(energy_array_spectrum,counts_array_spectrum)
    interpolated_counts=[]
    for i in range(len(energy_array)):
        interpolated_counts.append(interpolated_counts_function(energy_array[i]))

    # finally calculating the scatter continuum
    Scatter_continuum=[]
    for i in range(len(energy_array)):
        # weighted averaging
        sigma_by_rho_total=0
        mu_by_rho_total=0
        for ele in range(no_elements):
            sigmabyrho_interpolation_function_element=sigmabyrho_interpolated(atomic_number[ele])
            mubyrho_interpolation_function_element=mubyrho_interpolated(atomic_number[ele])
            sigma_by_rho_total+=(abundances[ele]/100)*sigmabyrho_interpolation_function_element(energy_array[i])
            mu_by_rho_total+=(abundances[ele]/100)*mubyrho_interpolation_function_element(energy_array[i])
        
        numerator_scattercontinuum=(interpolated_counts[i]*sigma_by_rho_total)
        denominator_scattercontinuum=(mu_by_rho_total*(cosec(solar_angle)+cosec(emis_angle)))
        Scatter_continuum.append(numerator_scattercontinuum/denominator_scattercontinuum)

    return energy_array, Scatter_continuum


initial_abundance=find_weighted_init(v0_lat,v1_lat,v2_lat,v3_lat,v0_lon,v1_lon,v2_lon,v3_lon)

# Write the values to a file
print("these are initial abundances", initial_abundance)
with open('abundances.txt', 'w') as file:
    for value in initial_abundance:
        file.write(f"{value}\n")

print("Initial Abundances have been written to abundances.txt")





                                     


            



            












