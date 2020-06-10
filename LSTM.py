from pandas import read_excel
from matplotlib import pyplot
from pandas import DataFrame
from datetime import datetime
from pandas import read_csv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from math import sqrt
import pandas as pd
import numpy as np

#Implementation uses Long-Short Tem Memory Recurrent Neural Networks to make predictions
#Following code is sourced from below projects and configured for this model and dataset

#Currently does a 12 month ahead prediction

#See pages below
#https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
def shiftPandasValues(array,newValue,slidingSize):
	for i in range(len(array) - 1):
		array[i] = np.copy(array[i + 1])

	array[len(array) - 1][0] = newValue[0]

	return array


# convert series to supervised learning
def preprocess_timeseries(data, months_lag=1, future_steps=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(months_lag, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, future_steps):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

runBenchmark = 1
twoFeaturesOnly = 0
dataFileName = 'Master.xlsx'
sheetName = 'SwitchingColumns'
outfolder = ''

if runBenchmark == 1:
	dataFileName = 'BenchmarkData\\SEClassProjects.xlsx'
	sheetName = 'RemovedWords'
	outfolder = 'BenchmarkData\\'


series = read_excel(open(dataFileName,'rb'), sheet_name=sheetName)

epochs = 100
train_percent = 0.6
forecast_horizon = 12
sliding_window_size = 5 # number for preceding time steps before prediction step
n_features = len(series.columns) - 1

if twoFeaturesOnly == 1:
	n_features = 2
	series.drop(series.iloc[:, (n_features + 1):], inplace = True, axis = 1) 

n_futuresteps = 1 #network output. Can explore further, if setting this to more than one will it predict multi-step-ahead?
n_train_months = round(series.shape[0] *  train_percent) # how many samples to use for training. Rest will be for test
currentFeatureName = series.columns[1]
x = series.values[:,1:len(series.columns)]

i = 0 #counter for graph names

# plot each feature in own graph
pyplot.figure()
for group in series.columns:
    if 	group != 'Date' and i < n_features:
        pyplot.plot(x[:,i])
        pyplot.title(group, y=0.5, loc='right')
        pyplot.savefig(outfolder + "Out\\Features\\" + str(i) + "_"+ group + ".png",bbox_inches='tight',dpi=100)
        pyplot.clf()
        i += 1



# ensure all data is float
x = x.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(x)

# preprocess data into samples using sliding window and step ahead number
reframed = preprocess_timeseries(scaled, sliding_window_size, n_futuresteps)
reframed.to_excel(outfolder + r'Out\\Preprocessed_Data.xlsx', index = False)
print(reframed.shape)

# split into train and test sets
values = reframed.values

train = values[:n_train_months, :]
test = values[n_train_months:, :]
# split into input and outputs
n_obs = sliding_window_size * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], sliding_window_size, n_features))
test_X = test_X.reshape((test_X.shape[0], sliding_window_size, n_features))

# design network
model = Sequential()
model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2]))) #100 number of neurons
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=epochs, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history


pyplot.plot(history.history['loss'])
# pyplot.plot(history.history['val_loss'], label='Cross Validation Loss')
pyplot.legend()
pyplot.xlabel('Epochs', color='#1C2833')
pyplot.ylabel('Loss', color='#1C2833')
pyplot.title('Cost Function Loss (MSE) During Training')
pyplot.savefig(outfolder+ "Out\\LossValuesDuringTraining.png",bbox_inches='tight',dpi=100)  
pyplot.show()

# make predictions into the future using horizon
yhat = model.predict(test_X) # predict first and take value as actual. Add to history
originalTestSamples = len(test_X)
originalTrainSamples = len(train_X)
#duplicate last sample from test set
predictedValues = np.array([yhat[-1]])

# test_X[-1] = np.roll(test_X[-1], 1, axis=0)

# print(test_X[-1])
# print('------------------')
for x in range(forecast_horizon):
	#duplicate last sample set
	temp = np.copy(test_X[-1])
	# print(temp)
	
	temp = shiftPandasValues(temp,yhat[-1], sliding_window_size)
	#shift data from sample, remove first, add new to last
	#temp = np.roll(temp, 1, axis=0)
	# print(temp)
	# print('------------------')
	#test_X[-1][sliding_window_size - 1][0] = yhat[-1] #set predicted value as history for next prediction
	test_X = np.insert(test_X,len(test_X),temp,0)

	yhat = model.predict(test_X)
	predictedValues = np.append(predictedValues, yhat[-1])
	test_y = np.append(test_y,yhat[-1]) #add predicted value as actual for now
	# print(test_X[-1])
	# print('------------------')

yhat = model.predict(test_X)
predictedValues = np.append(predictedValues, yhat[-1])

print('All Future Predicted Values')
print('----------------------------------')
print(str(predictedValues))
print('----------------------------------')

test_X = test_X.reshape((test_X.shape[0], sliding_window_size*n_features))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -(len(series.columns) - 2):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -(len(series.columns) - 2):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

pyplot.clf()


pyplot.plot(inv_y, label='actual')
pyplot.plot(inv_yhat, label='predicted')
pyplot.xlabel('Test Samples', color='#1C2833')
pyplot.ylabel('Values', color='#1C2833')
pyplot.title('Actual vs Predicted Values ('+ str(originalTestSamples)+' Test Samples):' + currentFeatureName )
pyplot.legend()

pyplot.axvspan(originalTestSamples, len(test_y), alpha=0.5, color='yellow')
pyplot.savefig(outfolder + "Out\\ActualVsPredictedValues.png",bbox_inches='tight',dpi=100)  
pyplot.show()

inv_yhat_df = pd.DataFrame(inv_yhat)
inv_y_df = pd.DataFrame(inv_y)

# print out all actual vs predicted values (inclusive of horizon steps into future)
inv_yhat_df.to_excel(outfolder + r'Out\\Predicted.xlsx', index = False)
inv_y_df.to_excel(outfolder + r'Out\\Actual.xlsx', index = False)

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

writeAccuracyFile = open(outfolder + "Out\\TESTRMSE.txt", "w")
writeAccuracyFile.write('Training Samples:' + str(originalTrainSamples))
writeAccuracyFile.write('Test Samples:' + str(originalTestSamples))
writeAccuracyFile.write('Future Steps Predicted:' + str(len(test_y) - originalTestSamples))
writeAccuracyFile.write('Test RMSE: %.3f' % rmse)

writeAccuracyFile.close()

print('Test RMSE: %.3f' % rmse)
