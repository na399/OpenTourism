import json
import utm
from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)

GMAPS_KEY = 'your API key here'

json_data = open('data/busstops.json').read()
data = json.loads(json_data)

list_utm = []
list_latlon = []
list_lat = []
list_lon = []

for stop in data['features']:
    list_utm.append(stop['geometry']['coordinates'][0])

list_latlon = list(map(lambda x: utm.to_latlon(x[0], x[1], 32, 'V'), list_utm))

list_lat = list(map(lambda x: x[0], list_latlon))
list_lon = list(map(lambda x: x[1], list_latlon))

mean_lat = sum(list_lat)/len(list_lat)
mean_lon = sum(list_lon)/len(list_lon)

# base code from http://bokeh.pydata.org/en/latest/docs/user_guide/geo.html

map_options = GMapOptions(lat=56.197134, lng=10.680734, map_type="roadmap", zoom=14)

plot = GMapPlot(x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options)
plot.title.text = "Bus stops"

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
plot.api_key = GMAPS_KEY

source = ColumnDataSource(
    data=dict(
        lat=list_lat,
        lon=list_lon,
    )
)

circle = Circle(x="lon", y="lat", size=90, fill_color="blue", fill_alpha=0.8, line_color=None)
plot.add_glyph(source, circle)

plot.add_tools(PanTool())
output_file("out/bus_plot.html")
show(plot)
