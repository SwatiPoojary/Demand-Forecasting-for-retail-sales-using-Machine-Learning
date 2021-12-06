import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

data = pd.read_csv('..\data\ProcessedData.csv');
data = data.sort_values(['Year','Month'], ascending=[True,True])
cols_poundsdata = ['Date','Year','Month_Name','Month','Day','OrderDate']
data.drop(cols_poundsdata, axis=1, inplace=True)
values = data.values

data['lag1'] = data['Cost'].shift(1)

data.dropna(inplace=True)

# split into train and test sets
X = data.values

train, test = X[:-36, :], X[-36:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# persistence model
def model_persistence(x):
	return x

predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)

mse = mean_squared_error(test[:, -1], predictions)
mae = mean_absolute_error(test[:, -1], predictions)
mape = mean_absolute_percentage_error(test[:, -1], predictions)

print('The mean Squared Error of our forecasts for Baseline is {}'.format(round(mse,2)))
print('The Root Mean Squared Error of our forecasts for Baseline is {}'.format(round(np.sqrt(mse), 2)))
print('The Mean absolute error of our forecasts for Baseline is {}'.format(round(mae,2)))
print('The Mean absolute percentage error of our forecasts for Baseline is {}'.format(round(mape,4)))

pyplot.plot(test[:, -1], label='Expected')
pyplot.plot(predictions, label='Predicted')
pyplot.xlabel('Months')
pyplot.ylabel('Sales Price(£ thousands)')
pyplot.legend()
pyplot.show()