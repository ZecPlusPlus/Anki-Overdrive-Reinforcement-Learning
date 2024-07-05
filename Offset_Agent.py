import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from drive import Overdrive
import threading
from stable_baselines3 import PPO, DQN,A2C
from stable_baselines3.common.env_checker import check_env
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import pickle
import torch as th
from gym.wrappers import NormalizeObservation
from gym.wrappers.normalize import RunningMeanStd
#from vqc_agent import VQCModel
#from wrappers import ScalingObservationWrapper
class OverdriveEnv(gym.Env):

    def __init__(self, car):
        super(OverdriveEnv, self).__init__()

        self.car = car
        self.car.setLocationChangeCallback(self._location_change_callback)
        self.laptime = 0
        # Action space: 0 = speed up, 1 = speed down, 2 = offset(-64) , 3 = offset(-48)  , 4 = offset(-32)  , 5 = offset(-16)  , 6 = offset(0)  , 7 = offset(16), 8 = offset(32), 9 = offset(48) , 10 = offset(64)
        self.action_space = spaces.Discrete(12)
        self.lap_counter = 0
        # Observation space: Transistion time, offset, piece ,next_piece, Curve/straight
        self.observation_space = spaces.Box(

            low=np.array([0, -70, 17, 17, 0, 0, 0]),
            high=np.array([3, 70, 40, 40, 1, 5, 5]),
            dtype=np.float32

        )   
    
        self.current_speed = 250
        self.track_map = [33,57,18,23,36,39,20,18,34]                                #self.track_map = [33,40,18,20,36,39,18,17,34]
        self.rotation_pieces = [18,20,23]
        self.straight_pieces = [33,34,36,57,39]
        self.action_taken = np.array([2,2])
        self.epsilon = 1e-8

        self.global_time = time.perf_counter()
        # Initial state
        self.state = [3,self.car._delegate.offset, 34, 33 , 1, 2,2]
        self.start_time = time.perf_counter()
        
        self.max_steps = 100 # Maximum number of steps per episode
        self.current_steps = 0
        self.lock = threading.Lock()
        self.last_piece = None
        self.last_piece_time = None
        self.cumulative_time = 0.0 # Is it okay to cummulate the time and then divide it by pieces?
        self.previous_cumulative_time = None
        self.num_pieces = 0  # Variable to count the number of pieces
        self.speed_before = 250
    

    #Need to delete this
    def _location_change_callback(self, addr, location, piece, offset, speed, clockwise):
        current_time = time.perf_counter()

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
    
    def normalize(self, obs):
        low = self.observation_space.low
        high = self.observation_space.high
        if obs[0] == None: 
            obs[0] = 0.9 #This is needed bec sometimes in the start he doesnt get the first transistion time

        obs = (obs - low) / (high - low + self.epsilon)
        return obs
    
    
    def step(self, action):
        # Execute the action
        self.speed_before = self.car.speed
    
        if action == 0:  # speed 800 + offset -40
            new_speed = 800
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, -40)
        elif action == 1:  # speed 600 + offset -40
            new_speed = 600
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, -40)
        elif action == 2:  # offset -40
            new_speed = 400
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, -40)
        elif action == 3:  # speed 800 + offset 0
            new_speed = 800
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, 0)
        elif action == 4:  # speed 600 + offset 0
            new_speed = 600
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, 0)
        elif action == 5:  # offset 0
            new_speed = 400
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, 0)
        elif action == 6:  # speed 800 + offset 40
            new_speed = 800
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, 40)
        elif action == 7:  # speed 600 + offset 40
            new_speed = 600
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, 40)
        elif action == 8:  # offset 40
            new_speed = 400
            self.car.changeSpeed(new_speed, 15000)
            self.car.changeLane(new_speed, 800, 40) 
        elif action == 9:  # offset 40
            new_speed = 400
            self.car.changeSpeed(new_speed, 15000) 
        elif action == 10:  # offset 40
            new_speed = 600
            self.car.changeSpeed(new_speed, 15000)  
        elif action == 11:  
            new_speed = 800
            self.car.changeSpeed(new_speed, 15000)   

        '''
        if action == 0:
            self.car.changeSpeed(300,15000)
        if action == 1:
            self.car.changeSpeed(400,15000)
        if action == 2:
            self.car.changeSpeed(600,15000)
        if action == 3:
            self.car.changeSpeed(700,15000)
        if action == 4:
            self.car.changeSpeed(800,15000)
        '''

        done = False
        truncated = False
        old_piece = self.car.piece

        # New state
        while not self.car._delegate.flag:
            continue
             
        self.car._delegate.flag = False
        
    
        if self.car._delegate.Transistion_time is not None:
            reward = -self.car._delegate.Transistion_time
        else:    
            reward = 0
        current_piece = self.track_map[(self.car._delegate.track_counter-1)%9]
        next_piece = self.track_map[(self.car._delegate.track_counter)%9]
        #print("Current_piece:",current_piece)
        piece_type = 0
        if next_piece in self.straight_pieces:
            piece_type = 1
        self.state = [self.car._delegate.Transistion_time, self.car._delegate.offset, current_piece , next_piece , piece_type, self.action_taken[0],self.action_taken[1]]
        
        if (current_piece == 34) and (next_piece == 33) :
            current_time = time.perf_counter()
            self.lap_counter +=1
            print("Round:", self.lap_counter, "with Time", current_time - self.lap_time_start)
            #reward -= current_time - self.lap_time_start
            current_lap_timer = current_time - self.lap_time_start
            writer.add_scalar("Timer_lap/train",current_lap_timer, self.lap_counter)    
            done = True

        self.state = self.normalize(np.array(self.state))   
        current_time = time.perf_counter() - self.global_time    
        writer.add_scalar("Time/train",reward, current_time)   
        # Check if the episode is donelap_counter = current_time - 
        self.current_steps += 1
        print("Reward",reward,"Action:",action,"Before speed", self.speed_before, "Speed now", self.car.speed)   

        
        self.action_taken = np.roll(self.action_taken,1)
        self.action_taken[-1] = action
        return np.array(self.state, dtype=np.float32), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state to the initial state
        
        if self.car.speed == 0:
            
            self.speed_before = 250
            self.car.changeSpeed(self.speed_before, 15000)
        else: 
            self.speed_before = self.car.speed   
       
        #print(self.speed_before)

        self.state = [3,self.car._delegate.offset,34, 33 , 1,2,2]
        self.state = self.normalize(np.array(self.state))
        while self.car.speed < self.speed_before:
            continue

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
addr = "CF:45:33:60:24:69" #"C9:96:EB:8F:03:0B" DC:7E:B8:5F:BF:46 "CF:45:33:60:24:69" "CB:76:55:B9:54:67"
car = Overdrive(addr)  # Assuming the car connection class is available
env = OverdriveEnv(car)
# Wrapper for quantum agent, remove comment, when using VQC
# env = ScalingObservationWrapper(env) 
print("Env")

# Check the environment
#check_env(env)

# if using VQC, remove comment
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[256,256,256,256])

# Train the environment with PPO
model = DQN('MlpPolicy', env, verbose=1,learning_starts=200,train_freq=5,target_update_interval=30,learning_rate=0.00001,exploration_initial_eps=1,exploration_fraction=0.1,gamma=0.99,exploration_final_eps=0,buffer_size=5000,policy_kwargs=policy_kwargs)
#model = PPO('MlpPolicy',env,verbose=2,n_steps=9,learning_rate=0.0001,gamma=0.99,policy_kwargs=policy_kwargs,gae_lambda=0.95,batch_size=9)
#model = A2C('MlpPolicy',env ,verbose=2,n_steps=13,learning_rate=0.0001,gamma=0.99,policy_kwargs=policy_kwargs,gae_lambda=0.95)

model.learn(total_timesteps=5000,progress_bar = True)

#with open('interactions.pkl', 'wb') as f:
#    pickle.dump(env.interactions, f)


print("training finished")
# Save the model
model.save("overdrive_ppo")

#with open('interactions.pkl', 'rb') as f:
#    interactions = pickle.load(f)


# Load the model
model = DQN.load("overdrive_ppo")

# Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
