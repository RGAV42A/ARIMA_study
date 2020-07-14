import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn import metrics
################################################

## data preparation from xml
path='/home/ragav/my_python_files/folder_data/pln.xml'
tree=ET.parse(path)
root=tree.getroot()
### time interval
end=5500
start=end-70
Z=range(start,end)
## read data in time interval
df=pd.concat([pd.Series(root[1][1][i].attrib) for i in Z],axis=1,join='inner')
df=df.swapaxes(0,1)
pwe=pd.to_datetime(df['TIME_PERIOD'])
week_col=[pwe[i].strftime("%V") for i in range(len(pwe.values))] # check type
year_col=[pwe[i].strftime("%Y") for i in range(len(pwe.values))]
arrays=[year_col]+[week_col]
index = pd.MultiIndex.from_arrays(arrays, names=('Year', 'CW'))
dff = pd.DataFrame(df['OBS_VALUE'].values,index=index,columns=['OBS_VALUE'],dtype=np.float64) # from 'str' to 'np.float64' !!!
ts=pd.Series(list(dff['OBS_VALUE']),index=pd.to_datetime(df['TIME_PERIOD']))
### decompose data
decomposition=seasonal_decompose(ts,freq=5) # 5 day in week
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid
### plot
'''
xax=range(len(ts))
plt.subplot(411)
plt.scatter(xax,trend,label='Trend')
plt.legend(loc='best')
plt.subplot(412)
plt.scatter(xax,seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(413)
plt.scatter(xax,ts,label='Original')
plt.legend(loc='best')
#plt.subplot(414)
#fig = decomposition.plot()
plt.savefig('/home/ragav/mysite/static/images/fig.png',dpi=225)
'''
### Checking for Stationary
## log transform
ts_log=np.log(ts)
ts_log.dropna(inplace=True)
s_test=adfuller(ts_log,autolag='AIC')
print('Log transform stationary check p-value:',s_test[1])
### Take first difference
ts_log_diff=ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
#plt.subplot(411)
#plt.title('Trend removed plot with first order diference')
#plt.plot(ts_log_diff)
#plt.ylabel('First order log diff')
#plt.savefig('/home/ragav/mysite/static/images/fig.png',dpi=125)
s_test=adfuller(ts_log_diff,autolag='AIC')
print('First order difference stationary check p-value:',s_test[1])
### moving average smoothens the line
moving_avg=ts_log.rolling(12).mean()
## plot results
'''
xax=range(len(ts_log_diff))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3))
ax1.set_title('First order difference')
ax1.tick_params(axis='x',labelsize=7)
ax1.tick_params(axis='y',labelsize=7)
ax1.plot(xax,ts_log_diff)
xax=range(len(moving_avg))
ax2.plot(ts_log)
ax2.set_title('Log vs Moving Avg')
ax2.tick_params(axis='x',labelsize=7)
ax2.tick_params(axis='y',labelsize=7)
ax2.plot(moving_avg,color='red')
plt.tight_layout()
fig.savefig('/home/ragav/mysite/static/images/fig1.png',dpi=125)
'''
####  Autocorrelation Test
# ACF chart

fig1, (ax3,ax4) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey = False, figsize=(8,4))
fig1=sm.graphics.tsa.plot_acf(ts_log_diff.values.ravel(),lags=20,ax=ax3)  # squeeze  -> raboti i z ravel i z squeeze
### draw 95% confidence interval line
ax3.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax3.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax3.set_xlabel('Lags')
### PACF chart
fig1=sm.graphics.tsa.plot_pacf(ts_log_diff,lags=20,ax=ax4)
### draw 95% confidence interval line
ax4.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax4.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax4.set_xlabel('Lags')
plt.savefig('/home/ragav/mysite/static/images/fig2.png',dpi=125)

##### Build ARIMA Model and Evaluate
# build model
model=sm.tsa.ARIMA(seasonal,order=(3,0,0))
results_arima=model.fit(disp=-1)
ts_predict=results_arima.predict()
xax=np.arange(len(ts_predict))

## plot data
plt.subplot(311,autoscale_on='True')
plt.plot(xax,seasonal)
plt.title('ARIMA Prediction - order(3,0,0)')
plt.plot(xax,ts_predict,'r--',label='Predicted')
plt.xlabel('Days')
plt.ylabel('Exchange rate')
plt.legend(loc='best')
plt.savefig('/home/ragav/mysite/static/images/fig3.png',dpi=125, figsize=(8,4))
#Evaluate model
print('AIC:',results_arima.aic)
print('BIC:',results_arima.bic)
print('Mean Absolute Error:',metrics.mean_absolute_error(ts_log.values,ts_predict.values))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(ts_log.values,ts_predict.values)))
# check autocorrelation
print('Durbin-Wtson stats:',sm.stats.durbin_watson(results_arima.resid.values))
print(results_arima.summary())

print('~~~~~~~~~~')





