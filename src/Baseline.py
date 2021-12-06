import pandas as pd
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_percentage_error

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
mae = mean_absolute_error(test[:, -1], predictions)
mape = mean_absolute_percentage_error(test[:, -1], predictions)
print('The mean absolute Error of our forecasts for RandomForest is {}'.format(round(mae,2)))
print('The Mean absolute percentage error of our forecasts for SARIMAX is {}'.format(round(mape,4)))

# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()