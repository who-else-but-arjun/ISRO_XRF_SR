from qgis.core import QgsProject
from qgis.utils import iface
# Get the layer by its name or ID
layer_name = 'xrf_data'  # Replace with the name of your XRF data layer
layer = QgsProject.instance().mapLayersByName(layer_name)[0]  # Get the layer by name

# Check if the layer exists
if layer:
    heatmap_renderer = QgsHeatmapRenderer()
    # Set the attribute to be used for the heatmap intensity (replace 'field_name' with the actual field name)
    heatmap_renderer.setColorRamp(QgsStyle.defaultStyle().colorRamp('Turbo'))  # Set color ramp
    heatmap_renderer.setRadius(10)  # Set the radius of influence for each point
    heatmap_renderer.setWeightExpression('AL_WT')
    # Apply the Heatmap renderer to the layer
    layer.setRenderer(heatmap_renderer)
    # Refresh the layer to apply changes
    layer.triggerRepaint()
else:
    print("Layer not found!")
