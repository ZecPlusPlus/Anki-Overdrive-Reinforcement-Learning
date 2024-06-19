import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def load_and_concatenate(file_prefix, num_files):
    data_list = []
    for i in range(num_files):
        data = np.load(f'{file_prefix}_{i}.npy')
        data_list.append(data)
    return np.concatenate(data_list, axis=0)

num_files = 5 
states = load_and_concatenate('States', num_files)
next_states = load_and_concatenate('Next_States', num_files)
actions = load_and_concatenate('Actions', num_files)
rewards = load_and_concatenate('Rewards', num_files)
done_counter = load_and_concatenate('Done_Counter', num_files)


data = np.concatenate((states, next_states, actions, rewards, done_counter), axis=1)

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

