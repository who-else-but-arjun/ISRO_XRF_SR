# Get the layer by name
layer_name = 'xrf_data'
layer = QgsProject.instance().mapLayersByName(layer_name)

# Check if the layer exists
if layer and len(layer) > 0:
    layer = layer[0]  # Get the first layer (in case there are multiple layers with the same name)

    # Check if the layer is a vector layer
    if layer.type() == QgsMapLayer.VectorLayer:
        # Get the layer's renderer (for a heatmap, it should be a heatmap renderer)
        renderer = layer.renderer()

        # Ensure the renderer is a heatmap renderer (which it should be if it's a vector heatmap)
        if isinstance(renderer, QgsHeatmapRenderer):
            # Set the opacity of the heatmap layer (0.0 to 1.0)
            opacity = 0.5  # Set opacity to 50%
            layer.setOpacity(opacity)
            
            # Refresh the layer to apply changes
            layer.triggerRepaint()
            print(f"Opacity of {layer_name} set to {opacity*100}%.")
        else:
            print(f"The layer '{layer_name}' is not using a heatmap renderer.")
    else:
        print(f"The layer '{layer_name}' is not a vector layer.")
else:
    print(f"Layer '{layer_name}' not found.")

