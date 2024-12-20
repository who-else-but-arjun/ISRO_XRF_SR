# Super-Resolution of Lunar XRF Elemental Abundances

This repository contains code developed for super-resolution by deep spatial interpolation model of the mapping of lunar surface elemental abundances obtained from Chandrayaan-2 CLASS instrument XRF data. The detailed description of the project is avaiable in this report (https://drive.google.com/file/d/12J4m65JyD-cekRiMmBzjrCBWQKoxNWYq/view?usp=drive_link).

## Project Structure
### Main Components:
1. **file_initialisation.py**:
   - Initializes the 64 subregion CSV files for the Moon's surface.
   - For each subregion, computes the latitudes and longitudes of 2 km x 2 km pixels.
   - Extracts features from image tiles using a pre-trained ResNet50 model.
   - Populates the CSV files with selected features, mare/highland classifications, and metadata.

2. **top_k_features.ipynb**:
   - Analyzes feature importance from ResNet50 extractions.
   - Identifies the top 300 features for further processing based on variance thresholding.

3. **Final.py**:
   - Combines all parts of the project.
   - Constructs subgraphs for abundance maps.
   - Train the GNN-CNN combined model on these subgraphs.
   - Returns the final high resolution abundance maps.

### Prerequisites :
- Python 3.10+
- GPU with CUDA support (optional but recommended).
- Clone the repository:
   ```bash
   git clone https://github.com/who-else-but-arjun/ISRO_XRF_SR.git
   cd ISRO_XRF_SR
   ```
- Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
- **Shapefile for Mare Regions**: `LROC_GLOBAL_MARE_180.SHP` ([https://drive.google.com/file/d/12J4m65JyD-cekRiMmBzjrCBWQKoxNWYq/view?usp=drive_link](https://drive.google.com/drive/folders/1Gpget_fLbG4ElaxwJFbuMO90GMVLFbqo?usp=sharing)).
- **Lunar Surface Image**: `Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif` (https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif).
- **CSV Files**: Intermediate outputs from `file_initialisation.py`.

Place the downloaded files in their respective directories as outlined below:
- `LROC_GLOBAL_MARE_180.SHP`: `./Mare Classification/`
- `Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif`: Root directory.

```
.
├── file_initialisation.py                                   # Initialization and feature extraction.
├── top_k_features.ipynb                                     # Feature analysis and selection.
├── Final.py                                                 # Full pipeline integration.
├── Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif                
├── requirements.txt                                         # Dependency file.
├── Mare Classification/                                     # Directory for shapefiles.
        | - LROC_GLOBAL_MARE_180.DBF
        | - LROC_GLOBAL_MARE_180.PRJ
        | - LROC_GLOBAL_MARE_180.SHP
        | - LROC_GLOBAL_MARE_180.SHP.XML
        | - LROC_GLOBAL_MARE_180.SHX
        | - LROC_GLOBAL_MARE_README.TXT  
├── PART2Output.npz                                           # Stores updated subregions
├── graphs/
├── masks/     
└── regions/                                                  # Outputs of file_initialisation.py.
```

### Outputs:
- **regions CSVs**:
  - Files named `subregion_<i>_<j>.csv`.
  - Columns include:
    - `lat_center` and `lon_center`: Geographic coordinates of the pixel centers.
    - Elemental abundances: Features like `Fe`, `Ti`, `Ca`, etc.
    - `mareOrHighland`: Binary classification indicating whether the region is Mare or Highland.
    - Top 300 spatial features: Feature columns extracted from lunar rgb images and top 300 selected features based on variance thresholding.

- **Graphs and their corresponding masks**:
  - Structured inputs for graph neural network models.
  - Encodes spatial and feature-based relationships within subregions.
  - Graphs stored in `graphs\graphs_subregion_<i>_<j>\subgraph_{i}_{j}_{row_idx}_{col_idx}.pt`.
  - Masks for the graphs stored in `masks\masks_subregion_<i>_<j>\mask_{i}_{j}_{row_idx}_{col_idx}.pt`.

- **Final High-Resolution Maps**:
  - Enhanced elemental abundance data for each subregion.
  - Saved as updated `subregion_<i>_<j>.csv` files for mapping.

---

## Running the Project

### Step 1: Run file_initialisation.py
This script initializes and populates the subregion CSV files.
```bash
python PART1.py
```
### Step 2: Run the Final Code
The `final.py` script combines all components and generates the high-resolution abundance maps.
```bash
python Final.py --mode 1 <input_data.csv>
```
mode = 1 for population of initialised csv files with the input elemental abundances data.
```bash
python Final.py --mode 2 <i> <j> <num_iterations>
```
mode = 2 for create the graphs for each subregion and training and interpolation for final high resolution abundnaces mapping.

---
