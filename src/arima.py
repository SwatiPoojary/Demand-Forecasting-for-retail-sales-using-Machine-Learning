from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import plot_params
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from statsmodels.tsa.arima_model import ARIMA


Ordata = pd.read_csv('DataSortedPoundsData.csv');
cols_poundsdata = ['Date','Year','Month']
Ordata.drop(cols_poundsdata, axis=1, inplace=True)

Ordata['OrderDate']= pd.to_datetime(Ordata['OrderDate'])
Ordata.reset_index().groupby(pd.Grouper(freq='1D', key='OrderDate')).mean()
Ordata = Ordata.set_index('OrderDate')

X = Ordata["Cost"].values
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

test_results = adfuller(Ordata["Cost"].dropna())
print('ADF Statistic: %f' % test_results[0])
print('p value: %f' % test_results[1])

# Original Series
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(Ordata.Cost);
axes[0, 0].set_title('Original Series')
plot_acf(Ordata.Cost, ax=axes[0, 1])

test_results = adfuller(Ordata["Cost"].diff().dropna())
print('ADF Statistic after 1 differencing: %f' % test_results[0])
print('p value after 1 differencing: %f' % test_results[1])
# 1st Differencing
axes[1, 0].plot(Ordata.Cost.diff());
axes[1, 0].set_title('1st Order Differencing')
plot_acf(Ordata.Cost.diff().dropna(), ax=axes[1, 1])



plt.show()

# d = 1
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2)
axes[0].plot(Ordata.Cost.diff());
axes[0].set_title('1st Differencing PACF')
plot_pacf(Ordata.Cost.diff().dropna(), ax=axes[1])

plt.show()

#  p = 2
fig, axes = plt.subplots(1, 2)
axes[0].plot(Ordata.Cost.diff());
axes[0].set_title('1st Differencing ACF')
plot_acf(Ordata.Cost.diff().dropna(), ax=axes[1])

plt.show()

#  q = 2

# ///////////////////////////////////////////////
# ARIMA
import statsmodels.api as sm

model = sm.tsa.ARIMA(Ordata.values, order=(2,1,2))
model_fit = model.fit()
print(model_fit.summary())
# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()

data = Ordata["Cost"]
train = data[:400]
test = data[401:]

# Build Model
# model = ARIMA(train, order=(3,2,1))
model = sm.tsa.ARIMA(train, order=(4, 1, 2))
fitted = model.fit(disp=-1)

# Forecast
fc, se, conf = fitted.forecast(28, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Calculating Metrics
mape = np.mean(np.abs(fc - test.values)/np.abs(test.values))  # MAPE
print("Mean Absolute Percentage Error is %f" %mape)