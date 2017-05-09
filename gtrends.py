import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pylab import rcParams
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

df_gtrends = pd.read_csv('data/google-trends.csv', header=1, parse_dates=['Week'])

df_gtrends.columns = ['week', 'Copenhagen', 'Amsterdam']
#df_gtrends.week = df_gtrends.week.dt.date

df_gtrends = pd.melt(df_gtrends, id_vars=['week'], var_name='city', value_name='search')

def ts_analysis(city):
    global df_city 
    df_city = df_gtrends[df_gtrends.city == city]
    df_city = df_city.set_index(df_city.week) 
    df_city.search = pd.rolling_mean(df_city.search, window=10)
    df_city = df_city.dropna(subset=['search'])
    
    rcParams['figure.figsize'] = 11, 13

    decomposition = sm.tsa.seasonal_decompose(df_city.search, model='multiplicative')
    fig = decomposition.plot()
    plt.show()
    #plt.savefig('out/tf_'+city+'.png')
    
    
    pass

ts_analysis('Copenhagen')
#ts_analysis('Amsterdam')

