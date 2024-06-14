# Car Detection and Control System

## Overview

This repository contains various Python scripts and modules designed for the detection and control of cars in a simulated environment. The main functionalities include detecting green and red cars, controlling the car's movements, testing location callbacks, and implementing reinforcement learning agents to optimize driving policies based on different reward structures.

## File Descriptions

### `car_detection.py`
This script detects green and red cars to demonstrate the learning process.

### `controller.py`
This file implements a controller to steer the car based on the detection inputs and other parameters.

### `drive.py`
This script sends commands to the embedded hardware, effectively linking the software control logic to the physical car hardware.

### `Location_test.py`
This script allows the car to drive at a constant speed to test if the correct location callbacks are being received.

### `Prasentation_file.py`
This demonstration file runs the car detection and controls the movements of both cars in the environment, showcasing the detection and driving capabilities.

### `RL_agent_piece_policy.py`
This reinforcement learning agent calculates rewards based on the time it takes for the car to reach the next piece. Faster completion results in a reward, while longer durations result in penalties.

### `RL_agent_speedDQN.py`
This script uses a Deep Q-Network (DQN) to learn the driving policy. Rewards are given based on speed: the faster the speed (above 350), the higher the reward. Speeds below 350 incur penalties to encourage faster learning.

### `RL_agent_speedPPO.py`
This script uses Proximal Policy Optimization (PPO) to learn the driving policy. Similar to the DQN approach, rewards are based on speed, with higher speeds yielding greater rewards and speeds below 350 resulting in penalties to promote faster learning.

### `Offset_Agent.py`
This script is a tryout file which uses instead of right and left commands the offset. It has an 11 action space instead of 4 action space. 

### `Car_detection_laps.py`
This script uses the camera to track the cars(red and green) and if they finish a lap, a lap counter gets added up by one. 

 
## Reinforcement Model:

# Action Space: 

11 Actions (1 = Speed up, 2 = Speed down, 3-11 = Change Offset(-64) - (64) -> Offset is between -64(left) and +64(right) we split it into 9 parts.)

# Observation Space:

Speed(low = 0, high = 800), Location(low = 0, high = 47), Piece(low = 0, high = 36), Offset(low = -70, high = 70)

# Reward System: 

The time it takes the car to reach each new piece.

## Python Version

To run the script, ensure you have a virtual enviorment with python 3.9.19

```bash
python3.9 -m venv .venv
```

Ensure you are in your virtual enviroment
```bash
source .venv/bin/activate
```
## Getting Started

To run the scripts, ensure you have the necessary dependencies installed. You can install the required packages using:

```bash
pip install -r requirements.txt
```


## Bluepy Installation if pip install fails
This is the installation tutorial by the repository: https://github.com/IanHarvey/bluepy

### Dependencies

To run this project, you need several libraries, including `bluepy`, which requires an executable `bluepy-helper` to be compiled from C source. This is done automatically if you use the recommended `pip` installation method. Otherwise, you can rebuild it using the Makefile in the `bluepy` directory.

### Installing Dependencies

#### Debian-based Systems:

To install the current released version, use the following commands:

```sh
sudo apt-get install python-pip libglib2.0-dev
sudo pip install bluepy
```
On Fedora do:

```sh
$ sudo dnf install python-pip glib2-devel
```

For Python 3, you may need to use pip3:

```sh
$ sudo apt-get install python3-pip libglib2.0-dev
$ sudo pip3 install bluepy
```

If this fails you should install from source.
```sh
$ sudo apt-get install git build-essential libglib2.0-dev
$ git clone https://github.com/IanHarvey/bluepy.git
$ cd bluepy
$ python setup.py build
$ sudo python setup.py install
```

### Run the Presentation

Execute the multiprocessor.py file. After a few seconds the Camera pops up. Next Tap out of the window and type into the konsole ü. This executes the 2 cars. 
If you want to terminate the camera-popup press q in the window.After that q in the terminal. This terminates the car controll.