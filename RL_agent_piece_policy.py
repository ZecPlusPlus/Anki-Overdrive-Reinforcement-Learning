import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import threading
from controller import CustomOverdrive

lock = threading.Lock()

class OverdriveEnv(gym.Env):
    def __init__(self, car):
        super(OverdriveEnv, self).__init__()
        self.car = car
        self.action_space = spaces.Discrete(4)  # 4 actions
        self.observation_space = spaces.Box(low=np.array([180, 0, 0]), high=np.array([800, 36, 45]), dtype=np.float32)
        self.car.setLocationChangeCallback(self.locationChangeCallback)
        self.last_piece_time = time.perf_counter()
        self.current_speed = 250 
        self.time_to_next_piece = 0
        self.last_piece = None
        self.piece_timestamp = {}
        self.piece_minimum_interval = 0.2
        self.learning_period = 1000
        self.callback_queue = []

        # Start the location change callback in a separate thread
        self.callback_thread = threading.Thread(target=self.process_callbacks)
        self.callback_thread.daemon = True
        self.callback_thread.start()

    def process_callbacks(self):
        while True:
            if self.callback_queue:
                addr, location, piece, offset, speed, clockwise = self.callback_queue.pop(0)
                self._process_location_change(addr, location, piece, offset, speed, clockwise)
            time.sleep(0.01)  # Adjust the sleep time as necessary

    def locationChangeCallback(self, addr, location, piece, offset, speed, clockwise):
        self.callback_queue.append((addr, location, piece, offset, speed, clockwise))

    def _process_location_change(self, addr, location, piece, offset, speed, clockwise):
        if piece is None:
            return
        current_time = time.perf_counter()
        with lock:
            if self.last_piece != piece and (piece not in self.piece_timestamp or current_time - self.piece_timestamp[piece] > self.piece_minimum_interval):
                if self.last_piece_time is not None:
                    piece_time = current_time - self.last_piece_time
                    print(f"Time to reach piece {piece}: {piece_time:.2f} seconds.")
                self.last_piece_time = current_time
                self.last_piece = piece
                self.piece_timestamp[piece] = current_time  
                self.current_speed = speed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.car.changeSpeed(0, 0)
        time.sleep(1)
        self.car.changeSpeed(250, 200)  # Start at speed 250
        self.learning_period = 1000
        return np.array([self.current_speed, 0, 0], dtype=np.float32), {}

    def step(self, action):
        self.learning_period -= 1
        print(self.learning_period)
        if action == 0:
            self.car.changeLaneRight(self.current_speed, 200)
            print("changeLaneRight")
        elif action == 1:
            self.car.changeLaneLeft(self.current_speed, 200)
            print("changeLaneLeft")
        elif action == 2:
            self.car.changeSpeed(min(self.current_speed + 50, 750), 200)
            print("Increase Speed:", self.current_speed)
        elif action == 3:
            self.car.changeSpeed(max(self.current_speed - 50, 200), 200)
            print("Decrease Speed:", self.current_speed)

        time.sleep(1) 
        reward = -self.time_to_next_piece  # Reward is negative time to next piece (we want to minimize time)
        obs = np.array([self.current_speed, 0, 0], dtype=np.float32)  

        
        done = False  
        if self.learning_period == 0:
            done = True
        truncated = False 
        return obs, reward, done, truncated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.car.disconnect()

# Initialize the car and environment
addr = "CF:45:33:60:24:69" 
car = CustomOverdrive(addr)
env = OverdriveEnv(car)

# Check the environment
check_env(env)

# Define the model
model = PPO("MlpPolicy", env, verbose=1).learn(total_timesteps=1000)

# Train the model
model.learn(total_timesteps=1000)
print("Training finished")

# Save the model
model.save("ppo_overdrive")
