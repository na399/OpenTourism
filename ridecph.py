#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
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

from sklearn.model_selection import train_test_split
import math
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error, median_absolute_error
import xgboost as xgb

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn import tree

GMAPS_KEY = 'your API key here'

MAP = False
BDAY = False
WEATHER = False
ML = True
LASSO = False


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
        g.set(xlabel="Hour of the Day", ylabel="Number of Cyclists passing the Counter", ylim=(0,max_count+100))
    else:
        g.set(xlabel="Hour of the Day", ylabel="Number of Cyclists passing the Counter")
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

def plot_weather(df_withweather, cut):
    plt.rcParams['figure.figsize']=(10,8)
    plt.rcParams['font.size']=14
    sns.set_palette("muted")
    sns.set_style("whitegrid")
    g = sns.swarmplot(x='hour', y='count', hue=cut, data=df_withweather)
    g.set(xlabel="Hour of the Day", ylabel="Number of Cyclists passing the Counter")
    plt.show()
    
plot_weather(df_withweather, 'temp_cut')
plot_weather(df_withweather, 'wspd_cut')


# Correlation analysis
corrMatt = df_withweather[["dewptm","hum","pressurem","rain","snow","tempm","vism","wspdm","count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)

if ML:

    # Create dataset for ML
    df_conds = pd.get_dummies(df_withweather.conds)
    df_data_set = pd.merge(df_withweather, df_conds, left_index=True, right_index=True)
    df_data_subset = df_data_set[["hour","tempm","hum","wspdm","dewptm","pressurem","rain","snow","vism",
                                 "Clear","Drizzle","Fog","Light Drizzle",
                                 "Light Rain","Light Rain Showers","Light Snow",
                                 "Mist", "Mostly Cloudy", "Overcast", "Partly Cloudy",
                                 "Rain", "Rain Showers", "Scattered Clouds", "Snow",
                                 "count"]]
    df_data_subset.hour = df_data_subset.hour.astype(int)
    
    # Check missing values and fill in by interpolation
    df_data_subset.isnull().sum()
    df_data_subset['pressurem'] = df_data_subset['pressurem'].interpolate().astype(int)
    df_data_subset['vism'] = df_data_subset['vism'].interpolate().astype(int)
    df_data_subset.isnull().sum()
    
    labels = df_data_subset["count"]
    train = df_data_subset.drop(["count"], 1)
    
    # https://www.kaggle.com/currie32/a-model-to-predict-number-of-daily-trips
    # Train the model
    
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state = 2)
    
    #15 fold cross validation. Multiply by -1 to make values positive.
    #Used median absolute error to learn how many trips my predictions are off by.
    
    def scoring(clf):
        global X_train, y_train
        scores = cross_val_score(clf, X_train, y_train, cv=15, n_jobs=-1, scoring = 'neg_median_absolute_error')
        print (np.median(scores) * -1)
        
        
    rfr = RandomForestRegressor(n_estimators = 55,
                                min_samples_leaf = 3,
                                random_state = 2,
                                n_jobs = -1)
    scoring(rfr)
    
    gbr = GradientBoostingRegressor(learning_rate = 0.12,
                                    n_estimators = 150,
                                    max_depth = 8,
                                    min_samples_leaf = 1,
                                    random_state = 2)
    scoring(gbr)
    
    dtr = DecisionTreeRegressor(min_samples_leaf = 3,
                                max_depth = 8,
                                random_state = 2)
    scoring(dtr)
    
    abr = AdaBoostRegressor(n_estimators = 100,
                            learning_rate = 0.1,
                            loss = 'linear',
                            random_state = 2)
    scoring(abr)
    
    
    # XGBoost
    
    import warnings
    warnings.filterwarnings("ignore")
    
    random_state = 2
    params = {
            'eta': 0.15,
            'max_depth': 6,
            'min_child_weight': 2,
            'subsample': 1,
            'colsample_bytree': 1,
            'verbose_eval': True,
            'seed': random_state,
        }
    
    n_folds = 15 #number of Kfolds
    cv_scores = [] #The sum of the mean_absolute_error for each fold.
    early_stopping_rounds = 100
    iterations = 10000
    printN = 50
    fpred = [] #stores the sums of predicted values for each fold.
    
    testFinal = xgb.DMatrix(X_test)
    
    kf = KFold(len(X_train), n_folds=n_folds)
    
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d' % (i+1))
        Xtrain, Xval = X_train.iloc[train_index], X_train.iloc[test_index]
        Ytrain, Yval = y_train.iloc[train_index], y_train.iloc[test_index]
        
        xgtrain = xgb.DMatrix(Xtrain, label = Ytrain)
        xgtest = xgb.DMatrix(Xval, label = Yval)
        watchlist = [(xgtrain, 'train'), (xgtest, 'eval')] 
        
        xgbModel = xgb.train(params, 
                             xgtrain, 
                             iterations, 
                             watchlist,
                             verbose_eval = printN,
                             early_stopping_rounds=early_stopping_rounds
                            )
        
        scores_val = xgbModel.predict(xgtest, ntree_limit=xgbModel.best_ntree_limit)
        cv_score = median_absolute_error(Yval, scores_val)
        print('eval-MSE: %.6f' % cv_score)
        y_pred = xgbModel.predict(testFinal, ntree_limit=xgbModel.best_ntree_limit)
        print(xgbModel.best_ntree_limit)
    
        if i > 0:
            fpred = pred + y_pred #sum predictions
        else:
            fpred = y_pred
        pred = fpred
        cv_scores.append(cv_score)
    
    xgb_preds = pred / n_folds #find the average values for the predictions
    score = np.median(cv_scores)
    print('Median error: %.6f' % score)
    
    
    #Train and make predictions with the best models.
    rfr = rfr.fit(X_train, y_train)
    gbr = gbr.fit(X_train, y_train)
    
    tree.export_graphviz(rfr, out_file='out/rfr.dot')  
    
    rfr_preds = rfr.predict(X_test)
    gbr_preds = gbr.predict(X_test)
    
    #Weight the top models to find the best prediction
    final_preds = rfr_preds*0.325 + gbr_preds*0.331 + xgb_preds*0.344
    print ("hourly error of count:", median_absolute_error(y_test, final_preds))
    
    #Create a plot that ranks the features by importance.
    def plot_importances(model, model_name):
        importances = model.feature_importances_
        std = np.std([model.feature_importances_ for feature in model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]    
    
        # Plot the feature importances of the forest
        plt.figure(figsize = (8,5))
        plt.title("Feature importances of " + model_name)
        plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()
        
    # Print the feature ranking
    print("Feature ranking:")
    
    i = 0
    for feature in X_train:
        print (i, feature)
        i += 1
        
    plot_importances(rfr, "Random Forest Regressor")
    plot_importances(gbr, "Gradient Boosting Regressor")
    
    
    fig,(ax1,ax2)= plt.subplots(ncols=2)
    fig.set_size_inches(12,5)
    sns.distplot(y_test,ax=ax1,bins=50)
    sns.distplot(final_preds,ax=ax2,bins=50)
    
    plt.clf()
    ax = sns.regplot(x=y_test, y=final_preds)
    
    
if LASSO:
    
    # https://www.kaggle.com/apapiu/regularized-linear-models
    # LASSO model
    
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
    
    coef = pd.Series(model_lasso.coef_, index = X_train.columns)
    
    coef.sort_values().plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")