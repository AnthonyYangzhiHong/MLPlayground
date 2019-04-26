# importing basic libraries

import numpy as np

import pandas as pd

from pandas import Series, DataFrame

from sklearn.model_selection import train_test_split

#import test and train file

train = pd.read_csv('Train.csv')

train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)

test = pd.read_csv('test.csv')

# importing linear regressionfrom sklearn

from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

#splitting into training and cv for cross validation

X = train.loc[:,['Item_Weight','Item_Visibility','Outlet_Establishment_Year','Item_MRP']]

x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales)



#training the model

lreg.fit(x_train,y_train)

#predicting on cv

pred = lreg.predict(x_cv)

#calculating mse

mse = np.mean((pred - y_cv)**2)

coeff = DataFrame(x_train.columns)

coeff['Coefficient Estimate'] = Series(lreg.coef_)

print(coeff)

print("#####################")

print(lreg.score(x_cv, y_cv))
