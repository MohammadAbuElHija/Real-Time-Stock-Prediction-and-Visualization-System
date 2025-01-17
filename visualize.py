import json

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time

real_data = []
prediction = []
time_points = []

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots()
line_real = ax.plot([],[], label='Real Data')
line_pred = ax.plot([],[],label="Prediction")
ax.legend(loc="upper left")
plt.xlabel("Time")
plt.ylabel("Value")

plt.title("Real Vs Predicted Stock Value")


def update(frame):
    time_points.append(frame)
    real_data.append(frame[0])
    prediction.append(frame[1])
    line_real.set_data(time_points,real_data)
    line_pred.set_data(time_points,prediction)
    return line_real,line_pred

def run(frame):
    ani = FuncAnimation(fig,update,frames=frame,interval=1000*60,blit=True)
    plt.show()


def show():
    real_data = []
    predicted_data = []
    times = []
    with open("prediction_data.json",'r') as f:
        try:
            show_data = json.load(f)
        except:
            show_data = []
            f.close()
    for dic in show_data:
        predicted_data.append(dic["Prediction"])
        real_data.append(dic["Real"])
        times.append(dic["Time"])

    plt.figure(figsize=(10, 5))
    plt.plot(times, predicted_data, label='Prediction', color='blue')
    plt.plot(times, real_data, label='Real', color='red')
    plt.xlabel('Time Points')
    plt.ylabel('Value')
    plt.title('Prediction vs Real Data')
    plt.legend()
    plt.show()

show()

