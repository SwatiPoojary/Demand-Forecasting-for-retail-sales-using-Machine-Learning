from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_csv('..\data\DataSortedPoundsData.csv');
data = data.sort_values(['Year', 'Month'], ascending=[True, True])
data_values = data["Cost"].values
mid = round(len(data_values) / 2)
data_values_1, data_values_2 = data_values[0:mid], data_values[mid:]
mean1, mean2 = data_values_1.mean(), data_values_2.mean()
var1, var2 = data_values_1.var(), data_values_2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

test_results = adfuller(data["Cost"].dropna())
print('ADF Statistic: %f' % test_results[0])
print('p value: %f' % test_results[1])
if (test_results[1] < 0.05):
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

#  q = 2

data.isnull().sum()
cols_poundsdata = ['Date','Year','Month']
data.drop(cols_poundsdata, axis=1, inplace=True)
data['OrderDate'] = pd.to_datetime(data['OrderDate'])

data.reset_index().groupby(pd.Grouper(freq='MS', key='OrderDate')).mean()
data = data.set_index('OrderDate')
print(data.index)

d = range(1, 2)
p = q = range(0, 3)

pdq = list(itertools.product(p, d, q))
listARima = []
arima_dict = {};
for param in pdq:
    model = ARIMA(data.values, order=param)
    # enforce_stationarity = False,
    results = model.fit(disp=0)
    paramList = [];
    paramList.append(param)
    print(results.aic)
    listARima.append(results.aic)
    arima_dict[results.aic] = paramList
    print('ARIMA{} - AIC:{}'.format(param, results.aic))

param = arima_dict[min(listARima)][0]
print(param)


# 1,1,2 ARIMA Model
model = ARIMA(data.values, order=param)
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()

# Create Training and Test
data = data["Cost"]
train = data[:400]
test = data[401:]

# Build Model
model = ARIMA(train, order=param)
fitted = model.fit(disp=-1)

# Forecast
fc, se, conf = fitted.forecast(28, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Calculating Metrics
mape = np.mean(np.abs(fc - test.values) / np.abs(test.values))  # MAPE
print("Mean Absolute Percentage Error is %f" % mape)