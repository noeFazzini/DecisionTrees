import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from decisionTreeClassifier import DecisionTreeClassifier
from decisionTreeRegressor import DecisionTreeRegressor

### DECISION TREE CLASSIFIER ###
# get the data
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
data = pd.read_csv("iris.csv", skiprows=1, header=None, names=col_names)
# print(data)

# split data
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

#fit the model
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()

#test model 
Y_pred = classifier.predict(X_test) 
acc = accuracy_score(Y_test, Y_pred)
acc_ = acc*100
print(str(acc) + '-->'+ str(acc_) + '%')

### DECISION TREE REGRESSION ###

data1 = pd.read_csv("airfoil_noise_data.csv")

# data split
X1 = data1.iloc[:, :-1].values
Y1 = data1.iloc[:, -1].values.reshape(-1,1)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=.2, random_state=41)

# fit the model
regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=3)
regressor.fit(X_train1,Y_train1)
regressor.print_tree()

# test
Y_pred1 = regressor.predict(X_test1) 
mse = np.sqrt(mean_squared_error(Y_test1, Y_pred1))
print('mse: ' + str(mse))