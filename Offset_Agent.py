import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from drive import Overdrive
import threading
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env

from vqc_agent import VQCModel
from wrappers import ScalingObservationWrapper

class OverdriveEnv(gym.Env):
    def __init__(self, car):
        super(OverdriveEnv, self).__init__()

        self.car = car
        self.car.setLocationChangeCallback(self._location_change_callback)

        # Action space: 0 = speed up, 1 = speed down, 2 = offset(-64) , 3 = offset(-48)  , 4 = offset(-32)  , 5 = offset(-16)  , 6 = offset(0)  , 7 = offset(16), 8 = offset(32), 9 = offset(48) , 10 = offset(64)
        self.action_space = spaces.Discrete(11)

        # Observation space: speed, location, piece, offset
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0,-70]),
            high=np.array([800, 47, 36, 70]),
            dtype=np.float32
        )
        # Initial state
        self.state = [250, self.car.location, self.car.piece, self.car.offset]
        self.start_time = time.perf_counter()
        self.max_steps = 100 # Maximum number of steps per episode
        self.current_steps = 0
        self.lock = threading.Lock()
        self.last_piece = None
        self.last_piece_time = None
        self.cumulative_time = 0.0 # Is it okay to cummulate the time and then divide it by pieces?
        self.previous_cumulative_time = None
        self.num_pieces = 0  # Variable to count the number of pieces

    def _location_change_callback(self, addr, location, piece, offset, speed, clockwise):
        current_time = time.perf_counter()

        with self.lock:
            if piece != self.last_piece:
                print(piece, offset, speed)
                if self.last_piece is not None:
                    time_taken = current_time - self.last_piece_time
                    self.cumulative_time += time_taken  # Add to cumulative time
                    self.num_pieces += 1  # Increment the piece count
                    print(f"Time to reach piece {piece}: {time_taken:.6f} seconds .")
                self.last_piece = piece
                self.last_piece_time = current_time
            # Update the car's location and speed
            self.car.offset = offset
            self.car.location = location
            self.car.piece = piece
            self.car.speed = speed

    def step(self, action):
        # Execute the action
        if action == 0:
            new_speed = min(self.state[0] + 60, 800)
            self.car.changeSpeed(new_speed, 500)
            print("New Speed == ", new_speed)
        elif action == 1:
            new_speed = max(self.state[0] - 30, 250)
            self.car.changeSpeed(new_speed, 500)
            print("New Speed == ", new_speed)
        elif action in range(2, 11):
            offsets = [-64, -48, -32, -16, 0, 16, 32, 48, 64]
            offset = offsets[action - 2]
            print("Taken Offset==", offset)
            self.car.changeLane(500,500,offset)
        
        time.sleep(2.0)  # Wait for 5 seconds to let the car perform the action
        
        with self.lock:
            reward = self.car.speed / 800.0  # Normalize the speed to get a value between 0 and 1
            
            # Penalize if the speed is below a certain threshold
            if self.car.speed < 350:
                reward -= 0.5
            print(self.car.speed, "Reward: ", reward)
        
        '''
        # Calculate reward based on the current speed
        with self.lock:
            
            current_time = time.perf_counter()
            if self.last_piece_time is not None:
                time_taken = current_time - self.last_piece_time
                reward = 1.0 / time_taken  # Inverse of time taken to reach the next piece
                self.last_piece_time = current_time
            else:
                reward = 0.0  # No reward if no piece change has occurred
            print(self.car.speed, "Reward: ", reward)
        '''
        # New state
        self.state = [self.car.speed, self.car.location, self.car.piece, self.car.offset]

        # Check if the episode is done
        self.current_steps += 1
        done = False
        truncated = False

        # Example condition for done: every 100 steps
        if self.car.piece == 34:
            print("Episode finished")
            done = True

        # Example condition for truncated: if the car's speed is zero
        if self.car.speed <= 0:
            truncated = True

        return np.array(self.state, dtype=np.float32), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state to the initial state
        self.state = [250, self.car.location, self.car.piece, self.car.offset]
        self.car.changeSpeed(250, 100)
        self.start_time = time.perf_counter()
        self.current_steps = 0
        self.last_piece = None
        self.last_piece_time = None
        self.cumulative_time = 0.0  # Reset cumulative time
        self.previous_cumulative_time = None  # Reset the previous cumulative time
        self.num_pieces = 0  # Reset the piece count
        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode='human'):
        # Implement rendering if needed
        pass

# Usage example
addr = "CF:45:33:60:24:69" 
car = Overdrive(addr)  # Assuming the car connection class is available
env = OverdriveEnv(car)

# Wrapper for quantum agent, remove comment, when using VQC
# env = ScalingObservationWrapper(env) 

# Check the environment
check_env(env)

# if using VQC, remove comment
policy_kwargs = dict(
#     features_extractor_class = VQCModel,
)

# Train the environment with PPO
model = DQN('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100)
print("training finished")
# Save the model
model.save("overdrive_ppo")

# Load the model
model = DQN.load("overdrive_ppo")

# Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
