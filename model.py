# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('IRIS.csv')

print(dataset.columns)

print(dataset.head())

X = dataset[['sepal.length','sepal.width','petal.length','petal.width']]

y = dataset[['variety']]


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

#Fitting model with trainig data
lr.fit(X, y)


# Saving model to disk
pickle.dump(lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[2.5, 3.2, 3.5, 3.7]]))
