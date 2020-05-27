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

#Implementation uses Long-Short Tem Memory Recurrent Neural Networks to make predictions
#Left to do;
#	Generate test set for Fiji for prediction of tourism numbers
#	Test set will have tourism numbers as blanks with all other columns roughly the same
#	How many steps into the future using predicted values as input to the next
#	Add Oceania datasets
#	Shuffle train datasets

#Objective:
#	How many months till normalization?
#	Tourism numbers for coming year

#See pages below
#https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
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
 
series = read_excel('Master.xlsx')
# pyplot.plot(series[:,1])
# pyplot.title(series.columns[1], y=0.5, loc='right')
# pyplot.show()
x = series.values[:,1:20]
data = preprocessing.normalize(x, axis=1, norm='l1')  #Normalize input data
i = 0
# plot each column
pyplot.figure()
for group in series.columns:
    if(group != 'Date'):
       
        pyplot.plot(data[:,i])
        pyplot.title(group, y=0.5, loc='right')
        pyplot.savefig(str(i) + "_"+ group + ".png",bbox_inches='tight',dpi=100)
       
        pyplot.clf()
        i += 1

# integer encode direction
# encoder = LabelEncoder()
# values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
x = x.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(x)

# specify the number of lag months
n_months = 6
n_features = 19
# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_months = 178 #178 training data from other diseases and countries. 48 will be used for fiji
train = values[:n_train_months, :]
test = values[n_train_months:, :]
# split into input and outputs
n_obs = n_months * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
# train_X, train_y = train[:, :n_obs], train[:, 1]
# test_X, test_y = test[:, :n_obs], test[:, 1]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_months, n_features))
test_X = test_X.reshape((test_X.shape[0], n_months, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    
# design network
model = Sequential()
model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2]))) #100 number of neurons
model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')


# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=20, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig("ActualVsPredictedLoss.png",bbox_inches='tight',dpi=100)  
pyplot.show()



# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_months*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -18:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -18:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
