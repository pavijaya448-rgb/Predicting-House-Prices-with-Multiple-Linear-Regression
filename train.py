import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r"C:\Users\WELCOME\Downloads\Housing.csv\Housing.csv")
data.head()
data.info()
data.shape
data.columns
data.dtypes
data.describe()
data.isnull().sum()
data.drop_duplicates()

binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for i in binary_columns:
    data[i] = data[i].map({'yes':1, 'no':0})

data['furnishingstatus'].unique()
data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)
data = data.astype(int)
data.head()

x = data.loc[:, ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = data[['price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape
x_test.shape

model = LinearRegression()
model

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
y_predict

print('R2 Score :',r2_score(y_test, y_predict))
print('Mean Squared Error :',mean_squared_error(y_test, y_predict))
print('Mean Absolute Error :',mean_absolute_error(y_test, y_predict))

model.predict([[7000, 2, 3, 2, 1]])

Regression_plot = sns.regplot(x=y_test, y=y_predict, ci=None, color='g')

axis_1 = sns.distplot(y_test, hist=False, color='g', label='Actual Value')
axis_2 = sns.distplot(y_predict, hist=False, color='b', label='Fitted Value')

import joblib
joblib.dump(model,'model.pkl')
loaded_model = joblib.load('model.pkl')
