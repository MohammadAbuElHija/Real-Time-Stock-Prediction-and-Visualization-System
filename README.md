# Real-Time Stock Prediction and Visualization System

## Description
This project integrates real-time stock price data, machine learning predictions, and visualization in a seamless pipeline. It leverages Kafka for data streaming, a pre-trained Linear Regression model for predictions, and Python-based visualization tools to display live comparisons between real and predicted stock prices. The system is designed for financial analysis and demonstrates key features of data engineering, machine learning, and real-time analytics.

---

## Key Features

### Real-Time Stock Data Integration
- Utilizes the `yfinance` library to fetch real-time stock data.
- Streams stock data to a Kafka topic (`stocks-data`).

### Data Engineering
- Computes financial indicators such as:
  - **Simple Moving Average (SMA)**
  - **Exponential Moving Average (EMA)**
  - **Relative Strength Index (RSI)**
  - **Moving Average Convergence Divergence (MACD)**
- Ensures clean and preprocessed data for machine learning predictions.

### Machine Learning Model
- Implements Linear Regression to predict future stock prices.
- Trains the model on historical stock data, incorporating key financial features.
- Saves the trained model using `joblib` for deployment.

### Real-Time Visualization
- Displays real-time and predicted stock prices using Matplotlib.
- Includes dynamic updates for continuous monitoring of stock trends.

### Streamlit Integration
- Provides an interactive Streamlit dashboard for users to view real-time data, predictions, and analysis.

---

## Technologies Used

- **Python Libraries**:
  - `pandas` and `numpy` for data manipulation.
  - `yfinance` for stock data retrieval.
  - `scikit-learn` for machine learning.
  - `matplotlib` and `seaborn` for visualization.
  - `streamlit` for the interactive web interface.
- **Apache Kafka**:
  - Used for real-time stock data streaming and consumer-producer architecture.
- **Joblib**:
  - For model serialization and deployment.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Set Up the Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start Kafka**:
   - Ensure Apache Kafka is installed and running locally.

4. **Run the Producer**:
   ```bash
   python producer.py
   ```

5. **Run the Consumer**:
   ```bash
   python consumer.py
   ```

6. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run visualize.py
   ```

---

## Usage

### Real-Time Data Streaming
- `producer.py`: Fetches real-time stock prices from Yahoo Finance and streams them to Kafka.
- `consumer.py`: Consumes stock prices, generates predictions, and saves them for visualization.

### Visualization
- `visualize.py`: Continuously displays real vs predicted stock prices on a dynamic graph.

### Model Training
- `train_model.py`: Trains a Linear Regression model using historical stock data.

---

## Future Enhancements

1. **Model Improvement**:
   - Incorporate more advanced models such as LSTMs or ARIMA for time-series forecasting.
2. **Dashboard Features**:
   - Add user input options to change stock tickers and view historical trends.
3. **Scalability**:
   - Deploy the system on cloud platforms for larger-scale data streaming and analytics.

---

## Acknowledgments
- **Yahoo Finance** for real-time stock data.
- **Apache Kafka** for robust data streaming.
- **scikit-learn** for easy-to-use machine learning tools.

