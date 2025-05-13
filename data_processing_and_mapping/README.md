# HIGH RESOLUTION ELEMENTAL MAPPING ON LUNAR SURFACE

## Table of Contents

- [GNU Data Language](#GNU)
- [XSPEC](#xspec)
- [ML](#ml)
- [Mapping](#mapping)
- [Website](#website)

## GNU Data Language (GDL)

### Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

### About the Software

This project uses GNU Data Language (GDL) for processing and analyzing FITS (Flexible Image Transport System) files containing data on lunar surface elemental abundances. GDL is an open-source, IDL-compatible compiler that excels in data visualization and analysis.

### Key Features

- Comprehensive FITS file processing
- Advanced integration with astronomical analysis libraries
- Efficient batch data processing
- Robust data visualization capabilities

### Technologies Used

- [GDL (GNU Data Language)](https://github.com/gnudatalanguage/gdl)
- [NASA's HEASoft](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/)
- [Astropy](https://www.astropy.org/)

### Getting Started

#### Prerequisites

Ensure the following tools are installed on your system:

1. GDL (GNU Data Language)
2. Miniconda or Python (for additional utilities)
3. FITS processing tools (e.g., HEASoft)

#### Installation

1. Install GDL using your package manager:
   ```bash
   sudo apt install gnudatalanguage
   ```

### Usage

#### Automated Processing Workflow

1. Prepare Your Data:

   - Place FITS files for a specific date in `data/class/{date}` directory
   - Add the corresponding Good Time Interval (GTI) file to the main project directory

2. Configure the Processing:

   - Update the date in `for_gdl.py`
   - Ensure all necessary scripts are in place

3. Run the Processing Pipeline:

   ```bash
   # Make scripts executable
   chmod +x for_gdl.exp

   # Execute processing scripts
   python for_gdl.py
   ./for_gdl.exp
   python name.py
   ```

#### Manual GDL Script Execution

1. Launch GDL:

   ```bash
   gdl
   ```

2. Load and Run Processing Script:
   ```idl
   @combine_fits.pro
   combine_fits,'input_directory','start_datetime','end_datetime','output_directory'
   ```

### Notes

- Solar flare analysis includes B, C, and M class events
- The processing pipeline automates command generation based on GTI and GOES solar flare data

### Dependencies

#### Mandatory Dependencies

- Readline: Provides command-line editing
- zlib: Enables compressed file access
- GSL (GNU Scientific Library): Supports advanced mathematical operations

## XSPEC

### Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Pre-Requisites](#prerequisites)
- [Usage](#usage)

### About the Software

XSPEC is a software designed for the analysis of astronomical X-ray spectra. It enables users to fit theoretical models to observational data, providing insights into the physical properties of celestial objects.

XSM Data Analysis Software (XSMDAS) processes XSM data step by step to create usable results. It converts raw data (level-0) into intermediate data (level-1) and then into final products (level-2). These final products include calibrated spectra and light curves, which can be customized with different settings.

#### Features

- Model Fitting
- Wide Format Compatibility
- Statistical Tools

### Getting Started

#### Installation

XSPEC:

1. Prerequisites
   HEASoftPy and other packages require Python elements:
   Python 3.6 (or newer)
   pip
   AstroPy (v4.0.0 or newer)
   NumPy (v1.7.0 or newer)
   SciPy (v1.5.0 or newer)
   MatPlotLib

   In a linux based Enviornment(like Ubuntu):

   sudo apt-get -y install libreadline-dev
   sudo apt-get -y install libncurses5-dev # or "libncurses-dev"
   sudo apt-get -y install ncurses-dev
   sudo apt-get -y install curl
   sudo apt-get -y install libcurl4
   sudo apt-get -y install libcurl4-gnutls-dev
   sudo apt-get -y install xorg-dev
   sudo apt-get -y install make
   sudo apt-get -y install gcc g++ gfortran
   sudo apt-get -y install perl-modules
   sudo apt-get -y install libdevel-checklib-perl
   sudo apt-get -y install libfile-which-perl
   sudo apt-get -y install python3-dev # or "python-dev"
   sudo apt-get -y install python3-pip
   sudo apt-get -y install python3-setuptools
   sudo apt-get -y install python3-astropy # needed for IXPE
   sudo apt-get -y install python3-numpy # needed for IXPE
   sudo apt-get -y install python3-scipy # needed for IXPE
   sudo apt-get -y install python3-matplotlib # needed for IXPE
   sudo pip install --upgrade pip # NOT needed in Ubuntu 20.04

2. In Bourne shell variants (bash/sh/zsh):
   conda install astropy numpy scipy matplotlib pip
   export PYTHON=$HOME/miniconda3/bin/python3

3. Set the standard environment variables:
   export CC=/usr/bin/gcc
   export CXX=/usr/bin/g++
   export FC=/usr/bin/gfortran
   export PERL=/usr/bin/perl
   export PYTHON=/usr/bin/python3

4. Unset all FLAGS and conda alias variables
   unset CFLAGS CXXFLAGS FFLAGS LDFLAGS build_alias host_alias
   export PATH="/usr/bin:$PATH"

5. Configure
   cd heasoft-6.34/BUILD_DIR/
   ./configure > config.txt 2>&1

6. BUILD
   make > build.txt 2>&1

7. Install
   make install > install.txt 2>&1

8. Initialize the software
   export HEADAS=/path/to/your/installed/heasoft-6.34/(PLATFORM)
   source $HEADAS/headas-init.sh

XSMDAS:

1. Pre-requisites:
   OS: Linux/Unix (CentOS 7.0+, Ubuntu 14.04+, RHEL 6.5+, Fedora 20.0+, SLED 11.0+, OS X 10.13+)
   Compiler: gcc 4.4+ Installation instructions
   Unzip the installation package ch2_xsmdas_yyyymmdd_vn.mm.zip to desired directory:
   unzip ch2_xsmdas_yyyymmdd_vn.mm.zip

2. Download the installation package of XSMDAS and CALDB from
   https://pradan.issdc.gov.in/pradan/

3. Setting Environment variables:
   In the directory where xsmdas exists as a folder
   nano .bashrc
   (Add following lines to ~/.bashrc)
   export xsmdas= /xsmdas
   export PATH="$xsmdas/bin:$xsmdas/scripts:$PATH" 
   export LD_LIBRARY_PATH="$xsmdas/lib/":$LD_LIBRARY_PATH export PFILES="$PFILES:$xsmdas/pfiles"

   where to be replaced with the absolute path under which xsmdas directory resides.

4. Installation of libraries:
   cd $xsmdas
   ./InstallLibs

5. Installation of CALDB:
   Unzip the package ch2_xsm_caldb_yyyymmdd.zip to $xsmdas directory as:
   unzip ch2_xsm_caldb_yyyymmdd.zip -d $xsmdas

6. Once the libraries are installed compile XSMDAS with
   cd $xsmdas
   Make

### Pre-Requisites and Dependencies

xspec: Used for spectral fitting and modeling.
xraylib: For X-ray line calculations and form factor data.
astropy: For handling FITS files.
numpy: For numerical operations and array handling.
scipy: For interpolation and numerical methods.
geopandas: For geographic data handling.
shapely: For geometric operations (e.g., polygons).
matplotlib: For optional data plotting (can be uncommented for visualization).

### Usage

#### Automated Processing Workflow

1. Organize Files
   Ensure the file structure is correctly arranged in the BUILD_DIR:

   BUILD_DIR/combined: Contains the combined files and supporting data:
   L1_added_files_time/: Stores all GDL-combined files organized based on GTI and GOES data.
   good_time.txt: Lists the start and end times for all combined files.
   {date}/xsm/data/{year}/{month}/{date}/:
   calibrated/: Includes GTI, light curve (lc), and spectrum (pha) files.
   raw/: Contains raw data files (FITS, HK, SA)

   Note: Change the path in 3.BUILD_DIR(XSPEC_and_XSMDAS)\get_xrf_lines_V1.py according to where the file is stored in your device.

2. Initialize Settings
   Update file_num.txt to start from 1:
   This ensures the process begins with the first combined file.

   (Optional) Delete final_abundance.csv if it exists:
   This prevents old data from being appended to the new results.

3. Run the Workflow
   Execute the following commands in the BUILD_DIR directory:
   chmod +x FINAL.exp  
   chmod +x script.exp  
   ./FINAL.exp

##### Explanation of Codes:

FINAL.exp - Spawns the script.exp for as many times as there are the number of files in L1_added_files_time

Script.exp -
The script starts by opening a Python session in the terminal.
Executes MAIN_0.py to perform the initial setup or data preparation required for the next steps.
After completing MAIN_0.py, the script runs MAIN_1.py for further data processing or calculations.
The script continues by running MAIN_2.py to perform additional processing.
After executing MAIN_2.py, the Python session is closed to proceed with the next set of tasks outside Python.
The script runs the ftflx2tab command to convert flux data into a tabular format for further use.
The script opens a new Python session to run further Python scripts.
The script then runs MAIN_3.py, continuing the processing with more advanced calculations or analysis.
After running MAIN_3.py, the script checks the value in the Optimisation.txt file, which dictates the next steps based on previous results.
If conditions are met (based on the value in Optimisation.txt), the script runs MAIN_4.py to finalize the processing or output.

File Description:

MAIN_0.py
Reads the file_num.txt to find the file for which processing has to be done.It also writes the start and end time of observation of that file (maximum 96 sec) into time.txt which is used later

MAIN_1.py

Part 1:
Solar Spectrum and Model Training
Extracts files based on time from time.txt, generates solar spectrum using XSMDAS, fits with vapec+powerlaw model, and saves the model and spectrum to .xcm and spectrum.qdp.
Optionally plots fitting results.

Part 2:
Lunar Elemental Abundance Analysis
Initializes abundance weights for highland and mare regions.
Generates file paths, tracks processed files, and extracts spectral data to modelop.txt.
Analyzes region overlap with maria, computes weighted initial abundances.
Extracts header data (solar angle, coordinates, exposure) and validates exposure time.
Saves coordinates, static parameters, and elemental data to respective files.
Interpolates cross-section data and spectrum, calculates scattered continuum, and saves initial abundances to abundances.txt.

MAIN_2.py

Reads time.txt to extract the start and stop time for the data.
Dynamically generates the path to the .fits file based on extracted times.
Reads spectrum.qdp for spectral energy, counts, and errors.
Interpolates spectral data for further analysis.
Reads X-ray scattering cross-section data for elements (e.g., Fe, Ti, Ca).
Interpolates the data for use in scatter continuum calculation.
Using the interpolated spectrum and cross-section data, calculates the scatter continuum based on elemental abundances.
Writes the energy ranges and scatter continuum values to ftools_infile_scatter.txt.

MAIN_3.py

Extracts start and stop times from time.txt and generates paths for the relevant .fits files.
Reads the spectral data (energy, counts, error) from spectrum.qdp.
Uses xraylib to calculate XRF lines for the elements (Fe, Ti, Ca, Si, Al, Mg, Na, O).
Interpolates X-ray scattering cross-section data for each element.
Loads the X-ray spectrum and applies the xrf_localmodel to fit the data.
The model is trained using the solar angle, emission angle, and other static parameters from static_par_localmodel.txt.
The fitted model's parameters are used to calculate the elemental abundances of Fe, Ti, Ca, Si, Al, Mg, Na, and O.
The script calculates the chi-square value to assess the quality of the fit.
If the reduced chi-square is below a threshold (10), the model fitting process is repeated for further refinement.
Deletes unnecessary files (.arf, .pha, .fits) after processing to save disk space.
Writes the computed abundances and chi-square values to abundances.txt and chi_squared.txt.

MAIN_4.py

The script reads 8 coordinate values from coordinates.txt. If the file contains more or fewer than 8 values, an error is raised.
The script reads 8 abundance values from abundances.txt. It separates the first 7 values and normalizes the last value with the others, ensuring there are exactly 8 values.
The script reads the chi-squared value from chi_squared.txt and validates it as a float.
The script combines the coordinates, normalized abundances, and the chi-squared value into a single row.
If the chi-squared value is less than 5, the script appends the row of combined values to final_abundances.csv. The row is formatted as a comma-separated line.


## Mapping

### Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Installation](#installation)

### About the Project

This project Uses Quantum Geographic Information System (QGIS) software for Visualizing the Lunar Abundances on Lunar Albedo Basemap at good sub-pixel resolution. The task is automated using PyQGIS codes. QGIS an open-source GIS platform that is freely available and offers a range of features suitable for various types of geospatial data analysis and visualization.

#### Key Features

- Comprhensive CSV File processing
- Processes XRF Spectral data and plot XRF detections.
- Generates Heatmaps and Graduated maps for visualizing spatial distribution of elemental abundances
- Leverages python libraries for automation

#### Technologies Used

- [QGIS](https://www.qgis.org/)
- [PyQGIS](https://docs.qgis.org/3.34/en/docs/pyqgis_developer_cookbook/index.html)
- [Web Mapping Service-Lunaserv](https://lunaserv.lroc.asu.edu/)
- [Geospatial Data Abstraction Library (GDAL)](https://gdal.org/en/stable/)

### Getting Started

#### Installation

1. Install QGIS from its official website:

   ```URL:
   https://www.qgis.org/download/
   ```

2. Setup GDAL as an environment variable in PATH:
   ```PowerShell
   [System.Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files\QGIS 3.34.11\bin", [System.EnvironmentVariableTarget]::Machine)
   ```

#### Automated Processing Workflow

1. Prepare Your Data
   Use the script CSV_combine.py to combine multiple CSV files into a single dataset with updated headers.
   Convert the combined CSV file into the appropriate format for generating graduated maps using CSV_Conversion_for_graduated_maps.py.
   Convert the CSV file into a format suitable for creating heatmaps with CSV_Conversion_for_heatmaps.py.

2. Configure the Processing
   Open the built-in Python console in QGIS by navigating to the Plugins menu.
   Confirm that all necessary Python scripts are available in your working directory.
   Double-check that all file paths to the data and scripts are correct.

3. Run the Processing Pipeline
   Use Import_Basemap.py to import the Lunar Albedo Basemap into QGIS.
   Load the CSV file as an xrf_data layer using Import_CSV_Layer.py.
   Generate polygons from the points in the CSV file, grouping them based on their IDs with CSV_File_Processing_to_polygons.py.
   Apply a color gradient based on lunar abundance to create a graduated map using Polygons_to_graduated_map.py.
   Generate heatmaps from the data using set_heatmap_layer_without_opacity.py.
   Modify the opacity of the heatmap using set_opacity_of_heatmap.py.
   Finally, export all the layers using the Export.py script for further analysis or presentation.

## Website

### KEY Features

- 3D CGI Moon Orbital and Spatial Views: Interactive 3D visualization of the Moon with full orbital and spatial controls.
- LAT/LONG Mapping: Maps specific points on the lunar surface based on latitude and longitude coordinates, displaying elemental abundances at each point.
- Clickable Markers: Clickable markers on the Moon’s surface to show subpixel resolution and elemental abundance for each selected point.
- 3D and 2D Elemental Maps: View both 3D and 2D maps displaying the elemental composition across the entire Moon.

### Getting Started

#### Technologies Used

- Three.js: A JavaScript library for creating 3D graphics in the browser.
- NASA CGI Moon Toolkit: A toolkit for displaying realistic 3D models of the Moon.

#### Installation

1. Unzip the Project: Unzip the website.zip file to your preferred location.
2. Navigate to the Project Directory:
   cd website
   cd src
3. Initialize npm: Initialize npm in the project directory by running
   npm init -y
4. Install Required Dependencies: Install the necessary packages for the project:
   npm install parcel -g # Install Parcel bundler globally
   npm install three # Install Three.js for 3D rendering
   npm i -g json-server # Install JSON Server for API simulation

#### Running the project

1. To enable dynamic interaction with data, prepare your CSV file with the following column headers in the specified order:
   V0_LATITUDE, V0_LONGITUDE, V1_LATITUDE, V1_LONGITUDE, V2_LATITUDE, V2_LONGITUDE, V3_LATITUDE, V3_LONGITUDE, FE_WT, TI_WT, CA_WT, SI_WT, AL_WT, MG_WT, NA_WT, O_WT, chi2 (optional)
   Once your CSV file is ready:
   cd src/js # Navigate to the src/js directory in your project
   node csvToJson.js # Convert the CSV to JSON by running the following command

2. Start the Website: Open two terminal windows.
   In the first terminal, run:
   parcel ./src/index.html
   This will start the Parcel development server and open the website in your default browser.

3. Start the API: In the second terminal window, run:
   json-server output.json
