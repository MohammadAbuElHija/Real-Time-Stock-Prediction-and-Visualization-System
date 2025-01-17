import json
import random



# Generate stock data for 100 entries
data = []
for i in range(1):  # Simulating 100 stock data points
    data.append({'Predicted':216.47099733, 'Real':216.55999755859375})  # Example for 'AAPL'

# Save the simulated data to a JSON file
with open('prediction_data.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Sample stock data saved to stock_data.json")
