from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)

GMAPS_KEY = 'your API key here'

# base code from http://bokeh.pydata.org/en/latest/docs/user_guide/geo.html

map_options = GMapOptions(lat=55.687659, lng=12.566553, map_type="roadmap", zoom=13)

plot = GMapPlot(
    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options
)
plot.title.text = "Copenhagen"

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
plot.api_key = GMAPS_KEY 

source = ColumnDataSource(
    data=dict(
        lat=[55.6, 55.7, 55.8],
        lon=[12.5, 12.6, 12.7],
    )
)

circle = Circle(x="lon", y="lat", size=10, fill_color="blue", fill_alpha=0.8, line_color=None)
plot.add_glyph(source, circle)

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
output_file("bus_plot.html")
show(plot)
