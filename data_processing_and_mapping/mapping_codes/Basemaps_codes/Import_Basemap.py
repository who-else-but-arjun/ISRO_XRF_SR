# Import necessary QGIS modules
from qgis.core import QgsRasterLayer, QgsProject

# Define the file path of your .tif file
tif_file_path = "C:\mapping_final\Lunar_Albedo_Base_Maps\Basemap_1.tif"  # Replace with the actual path

# Load the raster layer
raster_layer = QgsRasterLayer(tif_file_path, 'My Raster Layer')

# Check if the layer is valid
if not raster_layer.isValid():
    print("Failed to load the raster layer!")
else:
    # Add the raster layer to the QGIS project
    QgsProject.instance().addMapLayer(raster_layer)

    print("Raster layer loaded successfully!")
