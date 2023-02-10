import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error


def splitdata():
    '''
    this function imports, fills nulls, and splits the data
    '''
    city=pd.read_csv('GlobalLandTemperaturesByCity.csv')
    t=city[city['City']=='Thiruvananthapuram']
    t=t.ffill()
    t['dt']=pd.to_datetime(t['dt'])
    t=t.set_index('dt')
    train = t[:'1947']
    val= t['1948':'1990']
    test = t['1991':]
    return train,val,test


def plots():
    '''
    This function has exploratory plots
    '''
    city=pd.read_csv('GlobalLandTemperaturesByCity.csv')
    t=city[city['City']=='Thiruvananthapuram']
    t=t.ffill()
    t['dt']=pd.to_datetime(t['dt'])
    t=t.set_index('dt')
    pd.plotting.autocorrelation_plot(t.AverageTemperature.resample('Y').mean())
    plt.show()
    t.AverageTemperature.resample('Y').mean().plot()
    plt.xlabel('Date')
    plt.ylabel('Temp')
    plt.title('Yearly average temp')
    plt.show()
    plt.figure(figsize = (12,4))
    t['AverageTemperature'].resample('M').mean().plot(alpha= .25,label='Montly')
    t['AverageTemperature'].resample('Y').mean().plot(label='Yearly')
    plt.xlabel('Date')
    plt.ylabel('Temp')
    plt.legend()
    plt.show()


def evaluate(target_var, val, yhat_df):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(mean_squared_error(val[target_var], yhat_df[target_var], squared=False), 5)
    return rmse

def plot_and_eval(train, val, target_var, yhat_df, title):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1, color='#377eb8')
    plt.plot(val[target_var], label='Validate', linewidth=1, color='#ff7f00')
    plt.plot(yhat_df[target_var], label='yhat', linewidth=2, color='#a65628')
    plt.legend()
    plt.title(title)
    rmse = evaluate(target_var, val, yhat_df)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()


def append_eval_df(model_type, target_var, val, yhat_df, eval_df):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var, val, yhat_df)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


def modeling(train,val, target_var):
    '''
    this function will do all forms of time series modeling and give a final dataframe of the models and their rmse
    '''
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    '''
    last observation
    '''
    last_amount=train['AverageTemperature'][-1:][0]
    yhat_df = pd.DataFrame({'AverageTemperature': [last_amount]},index=val.index)
    plot_and_eval(train, val, 'AverageTemperature', yhat_df, 'Last Observation')
    plt.show()
    eval_df = append_eval_df('last_observed_value', 'AverageTemperature', val, yhat_df, eval_df)
    '''
    simple mean
    '''
    avg_amount=round(train['AverageTemperature'].mean(),2)
    yhat_df = pd.DataFrame({'AverageTemperature': [avg_amount]},index=val.index)
    plot_and_eval(train, val, 'AverageTemperature', yhat_df, 'Simple Mean')
    plt.show()
    eval_df = append_eval_df('simple_mean', 'AverageTemperature', val, yhat_df, eval_df)
    '''
    rolling average
    '''
    period = 30
    rolling_avg = round(train['AverageTemperature'].rolling(period).mean().iloc[-1], 2)
    yhat_df = pd.DataFrame({'AverageTemperature': [rolling_avg],}, index = val.index)
    plot_and_eval(train, val, 'AverageTemperature', yhat_df, 'Rolling 30 Day Average')
    plt.show()
    eval_df = append_eval_df('rolling_avg', 'AverageTemperature', val, yhat_df, eval_df)
    '''
    Holt optimized
    '''
    col = 'AverageTemperature' 
    model = Holt(train[col], exponential=False, damped=True)
    model = model.fit(optimized=True)
    yhat_values = model.predict(start = val.index[0],
                              end = val.index[-1])
    yhat_df[col] = round(yhat_values, 2)
    plot_and_eval(train, val, 'AverageTemperature', yhat_df, 'Holt Optimized')
    plt.show()
    eval_df=append_eval_df('holts_optimized','AverageTemperature', val, yhat_df, eval_df)
    '''
    holt seasonal
    '''
    hst_avg_temp_fit3 = ExponentialSmoothing(train.AverageTemperature, seasonal_periods=365, trend='add', seasonal='add', damped=True).fit()
    yhat_df = pd.DataFrame({'AverageTemperature': hst_avg_temp_fit3.forecast(val.shape[0])},
                          index=val.index)
    plot_and_eval(train, val, 'AverageTemperature', yhat_df, 'Holt Seasonal')
    plt.show()
    eval_df=append_eval_df('holts_seasonal','AverageTemperature', val, yhat_df, eval_df)
    '''
    previous year
    '''
    yhat_df = pd.DataFrame(val['AverageTemperature'] + train['AverageTemperature'].diff(365).mean())
    yhat_df.index = val.index
    plot_and_eval(train, val, 'AverageTemperature', yhat_df, 'Previous Year')
    plt.show()
    eval_df = append_eval_df("previous_year", 'AverageTemperature', val, yhat_df, eval_df)
    
    eval_df=eval_df.sort_values('rmse')
    return eval_df
    

def test(train, val, test, target_var):
    '''
    runs model on test and plots it
    '''
    yhat_df = pd.DataFrame(test['AverageTemperature'] + train['AverageTemperature'].diff(365).mean())
    plt.figure(figsize=(12,4))
    plt.plot(train[target_var], color='#377eb8', label='train')
    plt.plot(val[target_var], color='#ff7f00', label='validate')
    plt.plot(test[target_var], color='#4daf4a',label='test')
    plt.plot(yhat_df[target_var], color='#a65628', label='yhat')
    plt.legend()
    plt.title('Average Temperature')
    plt.show()
    trmse=mean_squared_error(test['AverageTemperature'], yhat_df['AverageTemperature'], squared=False)
    print('FINAL PERFORMANCE OF MODEL ON TEST DATA')
    print(f'rmse-average temperature: {trmse}')


def forecast(train, val, test, target_var):
    '''
    This function predicts out 5 years and plots it
    '''
    pred_df=pd.DataFrame()
    yhat_df = pd.DataFrame(test['AverageTemperature'] + train['AverageTemperature'].diff(365).mean())
    for n in range(1,6):
        future=test['2012-10':]
        add_to_pred=train['AverageTemperature'].diff(365).mean()
        future.index=future.index+pd.DateOffset(years=n)
        future['AverageTemperature']=future['AverageTemperature']+add_to_pred
        pred_df=pd.concat([pred_df, future['AverageTemperature']], axis=0)
    pred_df=pred_df.rename(columns={0:'AverageTemperature'})
    
    plt.figure(figsize=(12,7))
    plt.plot(train[target_var], color='#377eb8', label='Train')
    plt.plot(val[target_var], color='#ff7f00', label='Validate')
    plt.plot(test[target_var], color='#4daf4a', label='Test')
    plt.plot(yhat_df[target_var], color='#a65628', label='yhat')
    plt.plot(pred_df[target_var], color='#984ea3', label='Forecast')
    plt.title('Forecast Average Temperature')
    plt.legend()
    plt.show()