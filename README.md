# LSTMRNN

This model uses Long-Short Tem Memory Recurrent Neural Networks to make predictions

# Items Left;

1	Generate test set for Fiji to predict feature 'Tourism Numbers'. Although dataset contains multiple features, the  implementation predicts a single feature at a time. For this we will leave tourism numbers blank while the other features remain intact. Initial predictions is pretty bad as the model has learned from other bigger countries such as China with massive tourism numbers. Dataset only contains Fiji for the Oceania region at the moment, will need to add the PNG, NZ and Aus dataset in the COVIDMaster.xslx file. One way to avoid bias would be to shuffle the placement of the other diseases and countries before Fiji.

2   Might need to add Fiji to training set (with tourism numbers) 

# What are our objectives:

1   How many months till tourism numbers normalize?
2	Predict numbers for the next year
3   Possibly predict GDP?

See pages below for original code

https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
