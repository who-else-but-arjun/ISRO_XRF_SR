import geopandas as gpd
import matplotlib.pyplot as plt

# Path to your shapefile
shapefile_path = "LROC_GLOBAL_MARE_180.shp"

# Read the shapefile using geopandas
gdf = gpd.read_file(shapefile_path)

# Plot the geometries
fig, ax = plt.subplots(figsize=(10, 10))

for geom in gdf.geometry:
    if geom.geom_type == 'Polygon':
        # If it's a Polygon, plot its exterior
        x, y = geom.exterior.xy
        ax.plot(x, y, color='blue', linewidth=1)
    elif geom.geom_type == 'MultiPolygon':
        # If it's a MultiPolygon, plot each individual Polygon's exterior
        for poly in geom.geoms:  # Access individual polygons using `geoms`
            x, y = poly.exterior.xy
            ax.plot(x, y, color='blue', linewidth=1)

# Set plot labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Shapefile Visualization')
plt.show()
