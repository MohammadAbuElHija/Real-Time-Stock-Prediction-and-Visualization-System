import time
import yfinance as yf
from kafka import KafkaProducer
import json

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def fetch_realtime_stock_data(ticker):
    stock = yf.Ticker(ticker)
    while True:
        data = stock.history(period='1d', interval='1m')
        # Convert the DataFrame to a dictionary (latest record only)
        latest_data = data.tail(1).to_dict(orient='records')[0]
        del latest_data['Dividends']
        del latest_data['Stock Splits']
        producer.send('stocks-data', value=latest_data)
        producer.flush()
        print(f"Sent to Kafka: {latest_data}")
        time.sleep(60)  # Sleep for 1 minute to fetch new data


fetch_realtime_stock_data('TSLA')
