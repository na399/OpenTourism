#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import utm
from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)
from bokeh import palettes
from datetime import date
import holidays
import seaborn as sns
import matplotlib.pyplot as plt

GMAPS_KEY = 'your API key here'


# Load the Excel file stats from bike counters
df_GoBike = pd.read_excel('data/GoBike-GSM-data-set-2-mini-sample.xlsx')

#TabletImei
#GeoPoint_Longitude
#GeoPoint_Latitude

# See what do we have
df_GoBike.head

# Check missing values and fill in by interpolation
df_GoBike.isnull().sum()

# How many bikes are there? Ans: 1577
df_GoBike.TabletImei.unique().shape[0]
df_GoBike.TabletImeiMod = df_GoBike.TabletImei % 256

# Assign colours
list_colors=palettes.viridis(256)
def get_color(i):
    global list_colors
    return list_colors[i]

df_GoBike['color'] = map(get_color, df_GoBike.TabletImeiMod)

# base code from http://bokeh.pydata.org/en/latest/docs/user_guide/geo.html

map_options = GMapOptions(lat=55.685328, lng=12.538853, map_type="roadmap", zoom=12)

plot = GMapPlot(x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options)
plot.title.text = "Copenhagen GoBike"

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
plot.api_key = GMAPS_KEY

source = ColumnDataSource(
    data=dict(
        lat=df_GoBike.GeoPoint_Latitude,
        lon=df_GoBike.GeoPoint_Longitude,
        color=df_GoBike.color,
    )
)

circle = Circle(x="lon", y="lat", size=5, fill_color="color", fill_alpha=0.8, line_color=None)
plot.add_glyph(source, circle)

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
output_file("out/GoBike_plot.html")
show(plot)
