import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def create_recursive_data(data, window_size, target_name):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data[target_name] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data


data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("time")
# ax.set_ylabel("CO2")
# plt.show()
target = "target"
window_size = 5
data = create_recursive_data(data, window_size, target)

x = data.drop([target, "time"], axis=1)
y = data[target]
train_size = 0.8
num_samples = len(x)
x_train = x[:int(num_samples*train_size)]
y_train = y[:int(num_samples*train_size)]
x_test = x[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print("R2: {}".format(r2_score(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# for i, j in zip(y_predict, y_test):
#     print("Prediction: {}. Actual value: {}".format(i, j))
# fig, ax = plt.subplots()
# ax.plot(data["time"][:int(num_samples*train_size)], data["co2"][:int(num_samples*train_size)], label="Train")
# ax.plot(data["time"][int(num_samples*train_size):], data["co2"][int(num_samples*train_size):], label="Test")
# ax.plot(data["time"][int(num_samples*train_size):], y_predict, label="Prediction")
# ax.set_xlabel("time")
# ax.set_ylabel("CO2")
# ax.legend()
# ax.grid()
# plt.show()

# Predict for new data
current_data = [380.5, 390, 390.2, 390.9, 391.3]
for i in range(10):
    print(current_data)
    prediction = reg.predict([current_data]).tolist()
    print("CO2 in week {} is {}".format(i+1, prediction[0]))
    current_data = current_data[1:] + prediction
    print("-------------------------------")
