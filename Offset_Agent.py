import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from drive import Overdrive
import threading
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
#from vqc_agent import VQCModel
#from wrappers import ScalingObservationWrapper

class OverdriveEnv(gym.Env):
    def __init__(self, car):
        super(OverdriveEnv, self).__init__()

        self.car = car
        self.car.setLocationChangeCallback(self._location_change_callback)
        self.laptime = 0
        # Action space: 0 = speed up, 1 = speed down, 2 = offset(-64) , 3 = offset(-48)  , 4 = offset(-32)  , 5 = offset(-16)  , 6 = offset(0)  , 7 = offset(16), 8 = offset(32), 9 = offset(48) , 10 = offset(64)
        self.action_space = spaces.Discrete(2)
        self.lap_counter = 0
        # Observation space: speed, location, piece, offset
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0,-70]),
            high=np.array([800, 47, 50, 70]),
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
    
    
    
    #Need to delete this
    def _location_change_callback(self, addr, location, piece, offset, speed, clockwise):
        current_time = time.perf_counter()

        with self.lock:
            if piece != self.last_piece:
                #print(piece, offset, speed)
                if self.last_piece is not None:
                    time_taken = current_time - self.last_piece_time
                    self.cumulative_time += time_taken  # Add to cumulative time
                    self.num_pieces += 1  # Increment the piece count
                    #print(f"Time to reach piece {piece}: {time_taken:.6f} seconds .")
                self.last_piece = piece
                self.last_piece_time = current_time
            # Update the car's location and speed
            self.car.offset = offset
            self.car.location = location
            self.car.piece = piece
            self.car.speed = speed

    def step(self, action):
        # Execute the action
        speed = int(800/100*15)
        if action == 0:
            self.car.changeSpeed(min(self.state[0]+speed,800),500)
        if action == 1:
            self.car.changeSpeed(max(250,self.state[0]-speed),500)
        
        '''
        if action == 0:
            new_speed = min(self.state[0] + 60, 800)
            self.car.changeSpeed(new_speed, 500)
            print("New Speed == ", new_speed)
        elif action == 1:
            new_speed = max(self.state[0] - 60, 250)
            self.car.changeSpeed(new_speed, 500)
            print("New Speed == ", new_speed)
        elif action in range(2, 11):
            offsets = [-64, -48, -32, -16, 0, 16, 32, 48, 64]
            offset = offsets[action - 2]
            print("Taken Offset==", offset)
            self.car.changeLane(500,500,offset)
        '''
            
        #print("Transisition_time", self.car._delegate.Transistion_time)
        
        '''
        with self.lock:
            reward = self.car.speed / 800.0  # Normalize the speed to get a value between 0 and 1
            
            # Penalize if the speed is below a certain threshold
            if self.car.speed < 350:
                reward -= 0.5
            #print(self.car.speed, "Reward: ", reward)
        '''
        done = False
        truncated = False
        '''
        if self.car._delegate.Transistion_time is None:
            reward = self.car.speed / 800.0
        else:
            reward = -self.car._delegate.Transistion_time      
        '''
        
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
        old_piece = self.car.piece

        # New state
        while not self.car._delegate.flag:
            continue
             
        self.car._delegate.flag = False

        self.state = [self.car.speed, self.car.location, self.car.piece, self.car.offset]
        reward = self.car.speed / 800.0
        
        if (old_piece == 34 and self.state[2] == 33) or (old_piece == 34 and self.state[2]==40) :
            current_time = time.perf_counter()
            self.lap_counter +=1
            print(self.lap_counter, current_time - self.lap_time_start)
            reward += 500/current_time
            writer.add_scalar("Reward/train",reward, self.lap_counter)    
            done = True
        
        # Check if the episode is donelap_counter = current_time - 
        self.current_steps += 1
        print("Reward",reward)   
        # Example condition for truncated: if the car's speed is zero
        if self.car.speed <= 0:
            #Offset <70 or >70
            truncated = True

        return np.array(self.state, dtype=np.float32), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state to the initial state
        self.state = [250, self.car.location, self.car.piece, self.car.offset]
        self.car.changeSpeed(250, 100)
        self.start_time = time.perf_counter()
        self.lap_time_start = time.perf_counter()
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
addr = "C9:96:EB:8F:03:0B" #"C9:96:EB:8F:03:0B" DC:7E:B8:5F:BF:46
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
model = DQN('MlpPolicy', env, verbose=1,learning_starts=1000,train_freq=10,target_update_interval=100,learning_rate=0.0001,exploration_initial_eps=1,exploration_fraction=0.25,gamma=0.9999,buffer_size=5000)
#model = PPO('MlpPolicy',env,verbose=1,n_steps=8,batch_size=32,learning_rate=0.00001,gae_lambda=0.9,gamma=0.9999,use_sde=False,ent_coef=0.004)
model.learn(total_timesteps=10000)



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
