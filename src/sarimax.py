from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pylab import rcParams
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from datetime import datetime

warnings.filterwarnings("ignore")

data = pd.read_csv('..\data\ProcessedData.csv');
data = data.sort_values(['Year','Month'], ascending=[True,True])
data_values = data["Cost"].values
mid = round(len(data_values) / 2)
data_values_1, data_values_2 = data_values[0:mid], data_values[mid:]
mean1, mean2 = data_values_1.mean(), data_values_2.mean()
var1, var2 = data_values_1.var(), data_values_2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

test_results = adfuller(data["Cost"].dropna())
print('ADF Statistic on original data: %f' % test_results[0])
print('p value: %f' % test_results[1])
if(test_results[1] < 0.05):
    print("The data set used is a Stationary Data")
else:
    print("The data set used is a Non-Stationary Data")

# Original Series

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(data.Cost);
ax[0, 0].set_title('Original Data')
plot_acf(data.Cost, ax=ax[0, 1])

test_results = adfuller(data["Cost"].diff().dropna())
print('ADF Statistic after 1 differencing: %f' % test_results[0])
print('p value after 1 differencing: %f' % test_results[1])
if(test_results[1] < 0.05):
    print("The data set used is a Stationary Data")
else:
    print("The data set used is a Non-Stationary Data")

# 1st Differencing
ax[1, 0].plot(data.Cost.diff());
ax[1, 0].set_title('1st Order Differencing')
plot_acf(data.Cost.diff().dropna(), ax=ax[1, 1])
plt.show()
# d = 1

# PACF plot of 1st differenced series
fig, ax = plt.subplots(1, 2)
ax[0].plot(data.Cost.diff());
ax[0].set_title('1st Differencing PACF')
plot_pacf(data.Cost.diff().dropna(), ax=ax[1])
plt.show()

#  p = 2
fig, ax = plt.subplots(1, 2)
ax[0].plot(data.Cost.diff());
ax[0].set_title('1st Differencing ACF')
plot_acf(data.Cost.diff().dropna(), ax=ax[1])
plt.show()

data.isnull().sum()
data['OrderDate']= pd.to_datetime(data['OrderDate'])

data.reset_index().groupby(pd.Grouper(freq='MS', key='OrderDate')).mean()
data = data.set_index('OrderDate')
data.drop(data.tail(1).index, inplace=True)
y = data['Cost']

y_plt = data['Cost'];

y_plt = y_plt.dropna()
y_plt = y_plt.drop_duplicates(keep='first', inplace=False)

rcParams['figure.figsize'] = 18,8

# p = d = q = range(0,2)
d = range(1,2)
p =  q = range(0,3)

pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

listARima = []
arima_dict = {};
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            sarima_model = sm.tsa.statespace.SARIMAX(y_plt,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_invertibility=False)
            # enforce_stationarity = False,
            results = sarima_model.fit(disp=0)
            paramList = [];
            paramList.append(param)
            paramList.append(param_seasonal)
            listARima.append(results.aic)
            arima_dict[results.aic] = paramList
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
print(min(listARima))
param1 = arima_dict[min(listARima)][0]
param2 = arima_dict[min(listARima)][1]
print(param1)
print(param2)

sarima_model = sm.tsa.statespace.SARIMAX(y_plt,
                                order=param1,
                                seasonal_order=param2,
                                enforce_invertibility=False)
# enforce_stationarity = False,
sarima_result = sarima_model.fit(disp=0)
print(sarima_result.summary().tables[1])

sarima_result.plot_diagnostics(figsize=(16, 8))
plt.show()

# print(results.summary())
prediction = sarima_result.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
prediction_ci = prediction.conf_int()
print(prediction_ci)

ax1 = y_plt['1986':].plot(label='Actual')
prediction.predicted_mean.plot(ax=ax1, label='Predicted', alpha=.7, figsize=(14, 7))
ax1.fill_between(prediction_ci.index,
                prediction_ci.iloc[:, 0],
                prediction_ci.iloc[:, 1], color='k', alpha=.2)
ax1.set_xlabel('Year')
ax1.set_ylabel('Sales Price(£ thousands)')
plt.legend()
plt.show()

y_forecasted = prediction.predicted_mean
y_actual = y_plt['2015-01-01':]

mse = mean_squared_error(y_actual, y_forecasted)
mae = mean_absolute_error(y_actual, y_forecasted)
mape = mean_absolute_percentage_error(y_actual, y_forecasted)

print('The mean Squared Error of our forecasts for SARIMAX is {}'.format(round(mse,2)))
print('The Root Mean Squared Error of our forecasts for SARIMAX is {}'.format(round(np.sqrt(mse), 2)))
print('The Mean absolute error of our forecasts for SARIMAX is {}'.format(round(mae,2)))
print('The Mean absolute percentage error of our forecasts for SARIMAX is {}'.format(round(mape,4)))

pred_forecast = sarima_result.get_forecast(steps = 1)
print(pred_forecast.summary_frame(alpha=0.10))

pred_forecast_ci = pred_forecast.conf_int()
print("Forecast for next month (Oct 2021): ",format(round(pred_forecast.predicted_mean,2)))

