import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

with open('stock_data.json','r') as f:
    data = json.load(f)

df = pd.DataFrame(data)


# data cleaning

df.dropna(inplace= True)


# feature for analysis

#SMA: simple moving avg
df['SMA_10'] = df['Close'].rolling(window=10).mean()


#EMA: exponintial moving avg
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

#RSI: relative strengh index
# Calculate the Relative Strength Index (RSI)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))


#MACD: Moving Average Convergence Divergence
# Calculate MACD
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']

# print(df.head(20))
print(df.columns)

# data cleaning

df.dropna(inplace= True)

#Linear Regression ---------------------------------------------------------------
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace= True)

x_labels = df[['Open','High','Low','Volume','SMA_10','EMA_10','MACD']]
x_labels.dropna()
x = np.array(x_labels)

y = df['Target']


x_train ,x_test, y_train, y_test = train_test_split(x,y,test_size=0.75)



model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# statistic test

mse = mean_squared_error(y_test,y_pred)

print(f'Mean Squared Error: {mse}')



joblib.dump(model,'MLmodel.joblib')
