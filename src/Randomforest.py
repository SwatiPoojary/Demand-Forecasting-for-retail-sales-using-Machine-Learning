from numpy import asarray
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

data = pd.read_csv('..\data\ProcessedData.csv');
data = data.sort_values(['Year','Month'], ascending=[True,True])
cols_poundsdata = ['Date','Year','Month_Name','Month','Day','OrderDate']
data.drop(data.tail(1).index, inplace=True)
data.drop(cols_poundsdata, axis=1, inplace=True)
values = data.values

data['lag1'] = data['Cost'].shift(6)
data['lag2'] = data['Cost'].shift(5)
data['lag3'] = data['Cost'].shift(4)
data['lag4'] = data['Cost'].shift(3)
data['lag5'] = data['Cost'].shift(2)
data['lag6'] = data['Cost'].shift(1)
data.dropna(inplace=True)
data = data.values

predictions = list()
train, test = data[:-36, :], data[-36:, :]
trainX, trainy = train[:, :-1], train[:, -1]

model = RandomForestRegressor(n_estimators=1000)
model.fit(trainX, trainy)
score = model.score(trainX, trainy);
print("Model Score: ",score)
# step over each time-step in the test set
for i in range(len(test)):
	# split test row into input and output columns
	testX, testy = test[i, :-1], test[i, -1]
	y_pred = model.predict([testX])
	predictions.append(y_pred)
	print('>expected=%.1f, predicted=%.1f' % (testy, y_pred))
# estimate prediction error
mae = mean_absolute_error(test[:, -1], predictions)
mse = mean_squared_error(test[:, -1], predictions)
mape = mean_absolute_percentage_error(test[:, -1], predictions)
print('The mean Squared Error of our forecasts for RandomForest is {}'.format(round(mse,2)))
print('The Root Mean Squared Error of our forecasts for RandomForest is {}'.format(round(np.sqrt(mse), 2)))
print('The mean absolute Error of our forecasts for RandomForest is {}'.format(round(mae,2)))
print('The Mean absolute percentage error of our forecasts for SARIMAX is {}'.format(round(mape,4)))

# plot expected vs predicted
pyplot.plot(test[:, -1], label='Expected')
pyplot.plot(predictions, label='Predicted')
pyplot.xlabel('Months')
pyplot.ylabel('Sales Price(£ thousands)')
pyplot.legend()
pyplot.show()

# construct an input for a new prediction
row = values[-6:].flatten()
# make a one-step prediction
yhat = model.predict(asarray([row]))
print("Forecast for next month (Oct 2021): ",yhat[0])
