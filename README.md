# LSTMRNN

This model uses Long-Short Tem Memory Recurrent Neural Networks to make predictions

# Description

Dataset is put together from 5 known diseases. For each disease we have tried to gather at least 2 countries that were affected together with relevant information such as disease numbers, medical infrastructure and most importantly for this study tourist values such as visitor arrivals or income from tourism spending.

The model takes in 19 features and predicts a univariate time series. For this example it tries to predict the tourist arrivals. The time series sliding window and step ahead are important.

Sliding window has been set to 4 with one step ahead prediction. During test, as soon as a prediction is made it is fed back into the train set as input. This is done 'horizon value' number of times - in this case 12.

The model can be improved further.

https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
