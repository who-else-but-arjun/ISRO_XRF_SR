# Import necessary QGIS and PyQt modules
from qgis.core import (QgsProject, QgsMapSettings, QgsMapRendererSequentialJob, 
                       QgsCoordinateReferenceSystem, QgsMapLayer)
from qgis.PyQt.QtCore import Qt, QSize
import os
import time

def export_qgis_map(output_path=None, crs='EPSG:4326', resolution=(800, 600), background_color=Qt.white):
    """
    Export QGIS project with all visible layers in correct drawing order.
    """
    print("1. Starting QGIS Map Export Process...")
    start_time = time.time()

    # Default output path
    if output_path is None:
        output_path = os.path.join(os.path.expanduser('~'), 'QGIS_Export.tif')
    
    # Get current map canvas and project
    canvas = iface.mapCanvas()
    project = QgsProject.instance()
    layer_tree = project.layerTreeRoot()

    # Create map settings
    map_settings = QgsMapSettings()
    
    # Collect layers in correct drawing order (bottom to top)
    layers = []
    for layer_node in layer_tree.findLayers():
        layer = layer_node.layer()
        
        # Validate layer
        if not layer or not layer.isValid():
            continue
        
        # Check layer visibility and type
        if layer_node.isVisible() and layer.type() in [QgsMapLayer.RasterLayer, QgsMapLayer.VectorLayer]:
            print(f"   Adding layer: {layer.name()} (Type: {layer.type()})")
            layers.append(layer)
    
    # Reverse layers to maintain correct drawing order
    #layers.reverse()

    try:
        # Configure map settings
        map_settings.setLayers(layers)
        map_settings.setExtent(canvas.extent())
        map_settings.setDestinationCrs(QgsCoordinateReferenceSystem(crs))
        map_settings.setOutputSize(QSize(resolution[0], resolution[1]))
        map_settings.setBackgroundColor(background_color)

        # Render map
        print("2. Starting map rendering...")
        renderer_job = QgsMapRendererSequentialJob(map_settings)
        renderer_job.start()
        renderer_job.waitForFinished()

        # Save image
        img = renderer_job.renderedImage()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = img.save(output_path, "TIFF")
        
        if success:
            end_time = time.time()
            print(f"3. Export successful: {output_path}")
            print(f"   Layers exported: {len(layers)}")
            print(f"   Export time: {end_time - start_time:.2f} seconds")
        else:
            print("ERROR: Failed to save image")

    except Exception as e:
        print(f"ERROR during export: {e}")

# Uncomment and modify for use
export_qgis_map(
    output_path="C:\\mapping_final\output.tif", 
    resolution=(16384,8192)
)