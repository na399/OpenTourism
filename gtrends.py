import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pylab import rcParams
plt.style.use('fivethirtyeight')

df_gtrends = pd.read_csv('data/google-trends.csv', header=1, parse_dates=['Week'])

df_gtrends.columns = ['week', 'Copenhagen', 'Amsterdam']

#df_gtrends.week = df_gtrends.week.dt.date

df_gtrends = pd.melt(df_gtrends, id_vars=['week'], var_name='city', value_name='search')

def ts_analysis(city):
    global df_city 
    df_city = df_gtrends[df_gtrends.city == city]
    df_city.search += 1
    df_city = df_city.set_index(df_city.week) 
    df_city.search = df_city.search.rolling(window=10, center=False).mean()
    df_city = df_city.dropna(subset=['search'])
    
    # rcParams['figure.figsize'] = 11, 13
    
    decomposition = sm.tsa.seasonal_decompose(df_city.search, model='multiplicative')
    #fig = decomposition.plot()
    #fig.axes[0].set_ylim([0,100])
    
    return decomposition

    
# Perform time series analysis on the datasets from two cities
CPH = ts_analysis('Copenhagen')
AMS = ts_analysis('Amsterdam')

# Plot the observed values
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(311)
ax1.plot(CPH.observed, label="CPH")
ax1.plot(AMS.observed, label="AMS")  
ax1.set_ylabel("observed index")
# add legends
plt.legend(bbox_to_anchor=(0.7, 1.02, 1., .102), loc=3,
           ncol=2, borderaxespad=0.)
# make these tick labels invisible
plt.setp(ax1.get_xticklabels(), visible=False)


# Plot the trends
ax2 = fig.add_subplot(312, sharex=ax1)
ax2.plot(CPH.trend)
ax2.plot(AMS.trend)
ax2.set_ylabel("main trend")
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), visible=False)

# Plot the seasonal trends
ax3 = fig.add_subplot(313, sharex=ax1)
ax3.plot(CPH.seasonal)
ax3.plot(AMS.seasonal)
ax3.set_ylabel("seasonal trend")

plt.show()
#plt.savefig('out/gtrends_bikerental.png')
