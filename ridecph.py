#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import utm
from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)
import datetime
import holidays
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

GMAPS_KEY = 'your API key here'

MAP = False
BDAY = False
WEATHER = False

# Load the Excel file stats from bike counters
df_bikes = pd.read_excel('data/counters/cykeltaellinger-2014.xlsx', header=10)

# Change columns' name to English and change Hour format from 0-1 to 0, 1-2 to 1, ...
df_bikes.columns = ['ID', 'name', 'spor', 'X', 'Y', 'date',
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# See what do we have
df_bikes.head()

# How many counters are there? Ans: 27
df_bikes.ID.unique().shape[0]

# How many streets? Ans: 9
df_bikes.name.unique().shape[0]

# Check missing values and fill in by interpolation
df_bikes.isnull().sum()
df_bikes[1] = df_bikes[1].interpolate().astype(int)
df_bikes[2] = df_bikes[2].interpolate().astype(int)
df_bikes[23] = df_bikes[23].interpolate().astype(int)
df_bikes.isnull().sum()

# Edit the date format
df_bikes.date = pd.to_datetime(df_bikes.date, format='%d.%m.%Y')


# Convert utm to latlon
def to_lat(X, Y):
    return utm.to_latlon(X,Y,32,'U')[0]
def to_lon(X, Y):
    return utm.to_latlon(X,Y,32,'U')[1]

df_bikes['lat'] = map(to_lat, df_bikes.X, df_bikes.Y)
df_bikes['lon'] = map(to_lon, df_bikes.X, df_bikes.Y)

if MAP:


    list_lat = df_bikes.lat.unique().tolist()
    list_lon = df_bikes.lon.unique().tolist()

    # Plot where the counters are
    # base code from http://bokeh.pydata.org/en/latest/docs/user_guide/geo.html

    map_options = GMapOptions(lat=55.685328, lng=12.538853, map_type="roadmap", zoom=12)

    plot = GMapPlot(x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options)
    plot.title.text = "Copenhagen Bike Counters"

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

    circle = Circle(x="lon", y="lat", size=30, fill_color="blue", fill_alpha=0.8, line_color=None)
    plot.add_glyph(source, circle)

    plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
    output_file("out/gmap_plot.html")
    show(plot)



# Check whether date is business day or holiday
list_business = pd.bdate_range(df_bikes.date.min(), df_bikes.date.max())
list_holiday = holidays.DK()

def is_business(Date):
    global list_business
    return Date in list_business
def is_holiday(Date):
    global list_holiday
    return Date in list_holiday

df_bikes['business'] = map(is_business, df_bikes.date)
df_bikes['holiday'] = map(is_holiday, df_bikes.date)


def make_df_ID(ID, head=False, non_bday=False):

    global df_bikes
    if head:
        df_ID = df_bikes[df_bikes.ID == ID].head()
    else:
        df_ID = df_bikes[df_bikes.ID == ID]

    # We want to the number of cyclists in one column by having hours as rows instead "MELTING"
    df_ID = pd.melt(df_ID, id_vars=['ID', 'name', 'spor', 'X', 'Y', 'date', 'lat', 'lon', 'business', 'holiday'],
                    var_name='hour', value_name='count')

    global max_count
    max_count = df_ID['count'].max()

    if non_bday:
        df_ID = df_ID[(df_ID.holiday == True) | (df_ID.business == False)]

    return df_ID

def plot_(df_ID, ylim_max=False):
    plt.rcParams['figure.figsize']=(10,8)
    plt.rcParams['font.size']=20
    sns.set_style("whitegrid")
    g=sns.swarmplot(x='hour', y='count', data=df_ID)
    if ylim_max:
        g.set(xlabel="Hour within a Day", ylabel="Number of Cyclists passing the Counter", ylim=(0,max_count+100))
    else:
        g.set(xlabel="Hour within a Day", ylabel="Number of Cyclists passing the Counter")
    pass

if BDAY:
    # Let's deal with the busiest street first 'Torvegade'
    df_Torvegade = make_df_ID('101 1017592-0 0/ 635 T', non_bday=False)
    plot_(df_Torvegade)
    df_Torvegade_non_bday = make_df_ID('101 1017592-0 0/ 635 T', non_bday=True)
    plot_(df_Torvegade_non_bday, ylim_max=True)

    # Let's go to somewhere tourists don't go to 'Valby'

    # df_Vigerslev = make_df_ID('101 1018308-0 1/ 516 -', head=True)
    # plot_(df_Vigerslev)

if WEATHER:
    weather_day = {}
    df_weather = pd.DataFrame()

    for filename in os.listdir('data/weather'):
        path = os.path.join('data/weather', filename)
        json_data = open(path).read()
        data = json.loads(json_data)
        date = filename[-13:-5]
        weather_day[date] = data['history']['observations']

        for hour in weather_day[date]:
            hour_ = hour['date']['hour']
            #mday_ = hour['date']['mday']
            #mon_  = hour['date']['mon']
            hour.pop('date')
            hour.pop('utcdate')
            #hour['hour'] = hour_
            #hour['mday'] = mday_
            #hour['mon']  = mon_
            #hour['date'] = date
            index_ = datetime.datetime.strptime(date+hour_, '%Y%m%d%H').strftime('%Y-%m-%d %H:00:00')
            hour['date_hour'] = index_
            df_holder = pd.DataFrame(hour, index=[index_])
            df_weather = pd.concat([df_weather, df_holder])

    df_weather.to_csv('data/weather'+date[0:4]+'.csv')
    df_weather.date_hour = pd.to_datetime(df_weather.date_hour, format='%Y-%m-%d %H:%M:%S')
else:
    df_weather = pd.read_csv('data/weather2014.csv')
    df_weather.date_hour = pd.to_datetime(df_weather.date_hour, format='%Y-%m-%d %H:%M:%S')

df_Torvegade_non_bday = make_df_ID('101 1017592-0 0/ 635 T', non_bday=True)
# plot_(df_Torvegade_non_bday)
df_ = df_Torvegade_non_bday


def get_date_h(date, hour):
    return date + pd.DateOffset(hours=hour)

df_['date_hour'] = map(get_date_h, df_.date, df_.hour)


df_withweather = pd.merge(df_, df_weather, on="date_hour")

df_withweather.tempm = df_withweather.tempm.astype(int)
df_withweather.tempm.describe()
df_withweather['temp_cut'] = pd.cut(df_withweather.tempm, [-10, 5, 15, 30])
df_withweather['wspd_cut'] = pd.cut(df_withweather.wspdm, [0, 10, 25, 55])

def plot_weather(df_withweather):
    plt.rcParams['figure.figsize']=(10,8)
    plt.rcParams['font.size']=14
    sns.set_palette("muted")
    sns.set_style("whitegrid")
    g1 = sns.swarmplot(x='hour', y='count', hue='temp_cut', data=df_withweather)
    g1.set(xlabel="Hour within a Day", ylabel="Number of Cyclists passing the Counter")
    plt.show()
    
    plt.clf()
    g2=sns.swarmplot(x='hour', y='count', hue='wspd_cut', data=df_withweather)
    g2.set(xlabel="Hour within a Day", ylabel="Number of Cyclists passing the Counter")
    plt.show()

plot_weather(df_withweather)