results=processing.runAndLoadResults("native:pointstopath",
        {'INPUT':'delimitedtext://file:///C:\\mapping_final\\ch2_graduatedmap.csv?delimiter=,&xField=longitude&yField=latitude',
        'CLOSE_PATH':True,
        'ORDER_EXPRESSION':'"id"',
        'NATURAL_SORT':False,
        'GROUP_EXPRESSION':'"id"',
        'OUTPUT':'TEMPORARY_OUTPUT'})
print(results)
layer_name = 'xrf_data'
layer = QgsProject.instance().mapLayersByName(layer_name)
layer = layer[0]
print(layer)
clipped_dem=results['OUTPUT']
print(clipped_dem)
results=processing.runAndLoadResults("native:joinattributestable",
         {'INPUT':clipped_dem,
          'FIELD':'id',
          'INPUT_2':layer,
          'FIELD_2':'id',
          'FIELDS_TO_COPY':[],
          'METHOD':1,
          'DISCARD_NONMATCHING':False,
          'PREFIX':'',
          'OUTPUT':'TEMPORARY_OUTPUT'
         }
)
clipped_dem=results['OUTPUT']
print(clipped_dem)
results=processing.runAndLoadResults("native:fixgeometries", 
        {'INPUT':clipped_dem,
         'METHOD':1,
         'OUTPUT':'TEMPORARY_OUTPUT'
        }
)
clipped_dem=results['OUTPUT']
print(clipped_dem)
results=processing.runAndLoadResults("qgis:linestopolygons",
        {'INPUT':clipped_dem,
        'OUTPUT':'TEMPORARY_OUTPUT'
        }
)

