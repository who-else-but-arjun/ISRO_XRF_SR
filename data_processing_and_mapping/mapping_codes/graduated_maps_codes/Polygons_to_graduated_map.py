from qgis.utils import iface
from qgis.core import QgsProject, QgsGraduatedSymbolRenderer, QgsSymbol

# Define the name of the layer you want to select
layer_name = "Polygons"

# Get the layer by name
layers = QgsProject.instance().mapLayersByName(layer_name)

# Check if the layer exists
if layers:
    layer = layers[0]  # Since mapLayersByName returns a list, we select the first match
    iface.setActiveLayer(layer)
    renderer = QgsGraduatedSymbolRenderer()
    renderer.setClassAttribute('Mg') 
    style = QgsStyle.defaultStyle()
    color_ramp = style.colorRamp('Turbo') 
    renderer.setSourceColorRamp(color_ramp)
    renderer.setMode(QgsGraduatedSymbolRenderer.Quantile)
    renderer.updateClasses(layer, 5) 
    layer.setRenderer(renderer)
    layer.triggerRepaint()
    
else:
    print(f"Layer with name '{layer_name}' not found.")
