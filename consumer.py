from datetime import datetime
import time
from kafka import KafkaConsumer
import json
import joblib
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import visualize


import yfinance as yf
import plotly.graph_objs as go



# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'stocks-data',  # Kafka topic name
    bootstrap_servers=['localhost:9092'],  # Kafka broker address
    auto_offset_reset='latest',  # Start consuming only new messages
    enable_auto_commit=False,  # Disable auto commit, control commits manually
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))  # Deserialize JSON messages
)
st.title('Real vs Predicted Stock Price')
model = joblib.load('MLmodel.joblib')
file_path = 'stock_data.json'

predection_file = 'prediction_data.json'

print("Listening for new stock price updates...")







# Consume messages and append each one to the file individually
for message in consumer:
    stock_data = message.value
    pred_dict = stock_data.copy()
    print(f"Received stock data: {stock_data}")
    pred_dict['SMA_10'] = float(stock_data['Close'])
    pred_dict['EMA_10'] = float(stock_data['Close'])
    pred_dict['MACD'] = float(0)
    del pred_dict['Close']
    arr = np.array([list(pred_dict.values())], dtype=float)
    prediction = model.predict(arr)
    print(type(prediction))
    cur_time = np.array(pd.to_datetime(datetime.now()))

    preduction_data = {'Prediction': prediction[0],'Real':stock_data['Close'], "Time":time.time()}

    real_price = float(stock_data['Close'])






    print(f'prediction: {prediction} real close price: {stock_data["Close"]}')
    with open(file_path,'r') as f1:
        try:
            data_json = json.load(f1)
        except json.JSONDecodeError:
            data_json = []

    with open(file_path,'w'):
        pass

    data_json.append(stock_data)

    # Write each stock data entry directly to the file
    with open(file_path, 'a') as f:
        json.dump(data_json, f,indent= 4)  # Dump only the current message
        f.write('\n')

    with open(predection_file, 'r') as f3:
        try:
            data_json1 = json.load(f3)
        except json.JSONDecodeError:
            data_json1 = []

    with open(predection_file, 'w'):
        pass

    data_json1.append(preduction_data)

    with open(predection_file, 'a') as f2:
        json.dump(data_json1, f2,indent= 4)  # Dump only the current message
        f2.write('\n')

    visualize.show()





