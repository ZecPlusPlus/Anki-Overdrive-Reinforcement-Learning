import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from drive import Overdrive
import gymnasium as gym
from gym import spaces
import numpy as np
from Offset_Agent import OverdriveEnv
import pennylane as qml
from pennylane.qnn import TorchLayer
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights_0, weight_1):
    qml.RX(inputs[0], wires=0)
    qml.RX(inputs[1], wires=1)
    qml.Rot(*weights_0, wires=0)
    qml.RY(weight_1, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

weight_shapes = {"weights_0": 3, "weight_1": 1}

qlayer = TorchLayer(qnode, weight_shapes)

print(qlayer)