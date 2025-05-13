from qgis.core import QgsVectorLayer, QgsProject

# Path to your CSV file
csv_file_path = "C:\mapping_final\ch2_data.csv"

# Create the layer with the appropriate URI
uri = f"file:///{csv_file_path}?delimiter=,&xField=longitude&yField=latitude"

# Replace 'longitude' and 'latitude' with the actual column names for your coordinates
csv_layer = QgsVectorLayer(uri, "xrf_data", "delimitedtext")

# Check if the layer is valid and add it to the project
if csv_layer.isValid():
    QgsProject.instance().addMapLayer(csv_layer)
else:
    print("Layer failed to load!")
