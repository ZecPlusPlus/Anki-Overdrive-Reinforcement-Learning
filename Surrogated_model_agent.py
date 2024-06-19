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
        self.action_space = spaces.Discrete(9)
        self.lap_counter = 0
        # Observation space: speed, piece, offset
        self.observation_space = spaces.Box(
            low=np.array([0, 0,-70, 0, -3.5]),
            high=np.array([900,  50, 70, 4, 0]),
            dtype=np.float32
        )
        self.action_taken = 2
        self.epsilon = 1e-8
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.global_time = time.perf_counter()
        # Initial state
        self.state = [250, self.car.piece, self.car.offset, 2, -3.0]
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
        
        self.state_before=[]
        self.state_after=[]
        self.action_list=[]
        self.rewards_list=[]
        self.done_list =[]
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
    
    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def step(self, action):
        # Execute the action
        self.state_before.append(self.state)
        self.speed_before = self.car.speed
        #Left
        if action == 0:
            speed_minus25 = int(self.speed_before-800/100*25)
            self.car.changeSpeed(max(300,speed_minus25),500)
            self.car.changeLaneLeft(self.car.speed,500)
        if action == 1:
            speed_plus25 = int(self.speed_before+ 800/100*25)
            self.car.changeSpeed(min(700,speed_plus25),500)
            self.car.changeLaneLeft(self.car.speed,500)
        if action == 2:
            self.car.changeSpeed(self.speed_before,500)
            self.car.changeLaneLeft(self.car.speed,500)
        #Right    
        if action == 3:
            speed_minus25 = int(self.speed_before - 800/100*25)
            self.car.changeSpeed(max(300,speed_minus25),500)
            self.car.changeLaneRight(self.car.speed,500)
            
        if action == 4:
            speed_plus25 = int(self.speed_before + 800/100*25)
            self.car.changeSpeed(min(700,speed_plus25),500)
            self.car.changeLaneRight(self.car.speed,500)
        if action == 5:
            self.car.changeSpeed(self.speed_before,500)
            self.car.changeLaneRight(self.car.speed,500)
            
        #Keep    
        
        if action == 6:
            speed_minus25 = int(self.speed_before- 800/100*25)
            self.car.changeSpeed(max(300,speed_minus25),500)
        if action == 7:
            speed_plus25 = int(self.speed_before+ 800/100*25)
            self.car.changeSpeed(min(700,speed_plus25),500)
        if action == 8:
            self.car.changeSpeed(self.speed_before,500)
           
        

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
        
        self.state = [self.car.speed, self.car.piece, self.car.offset, self.action_taken , reward]
        
        if(old_piece == 17 or old_piece == 34):

            if self.state[1] == 33 or self.state[1]==40:
            
                current_time = time.perf_counter()
                self.lap_counter +=1
                print(self.lap_counter, current_time - self.lap_time_start)
                #reward -= current_time - self.lap_time_start
                current_lap_timer = current_time - self.lap_time_start
                writer.add_scalar("Timer_lap/train",current_lap_timer, self.lap_counter)    
                done = True

        self.state = self.normalize(np.array([self.state]))[0]    
        current_time = time.perf_counter() - self.global_time    
        writer.add_scalar("Time/train",reward, current_time)   
        # Check if the episode is donelap_counter = current_time - 
        self.current_steps += 1

        print("Reward",reward,"Action:",action,"Before speed", self.speed_before, "Speed now", self.car.speed,"Old_piece:",old_piece, "New_piece:",self.car.piece)   

        
        self.action_taken = action
        self.rewards_list.append(reward)
        self.action_list.append(action)
        self.state_after.append(self.state)
        self.done_list.append(done)
        return np.array(self.state, dtype=np.float32), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state to the initial state
        
        if self.car.speed == 0:
            
            self.speed_before = 250
        else: 
            self.speed_before = self.car.speed   
       
        
        self.car.changeSpeed(self.speed_before, 500)
        self.state = [self.car.speed,  self.car.piece, self.car.offset, 2 , -3.0]
        self.state = self.normalize(np.array([self.state]))[0]
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
addr = "DC:7E:B8:5F:BF:46" #"C9:96:EB:8F:03:0B" DC:7E:B8:5F:BF:46
car = Overdrive(addr)  # Assuming the car connection class is available
env = OverdriveEnv(car)
# Wrapper for quantum agent, remove comment, when using VQC
# env = ScalingObservationWrapper(env) 




# Check the environment
#check_env(env)

# if using VQC, remove comment
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(vf=[128,128,128],pi=[256, 256, 256]))

# Train the environment with PPO
#model = DQN('MlpPolicy', env, verbose=1,learning_starts=200,train_freq=5,target_update_interval=30,learning_rate=0.00001,exploration_initial_eps=1,exploration_fraction=0.25,gamma=0.99,buffer_size=5000,policy_kwargs=policy_kwargs)
#model = PPO('MlpPolicy',env,verbose=2,n_steps=9,learning_rate=0.0001,gamma=0.99,policy_kwargs=policy_kwargs,gae_lambda=0.95,batch_size=9)
model = A2C('MlpPolicy',env,verbose=2,n_steps=8,learning_rate=0.0001,gamma=0.99,policy_kwargs=policy_kwargs,gae_lambda=0.95)

#with open('interactions.pkl', 'wb') as f:
#    pickle.dump(env.interactions, f)


# Test the trained model
obs, _ = env.reset()

for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
upper_file = "_0"
np.save("States"+upper_file+".npy",np.array(env.state_before))        
np.save("Next_States"+upper_file+".npy",np.array(env.state_after))
np.save("Actions"+upper_file+".npy",np.array(env.action_list))
np.save("Rewards"+upper_file+".npy",np.array(env.rewards_list))
np.save("Done_Counter"+upper_file+".npy",np.array(env.done_list))
