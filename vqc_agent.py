from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from typing import Tuple, List
import torch

def create_VQC(num_qubits: int, num_layers: int) \
        -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
    """
    Creates a VQC with Rx rotations in the input encoding layer, with data
    re-uploading Ry and Rz rotations in the variational layer and circular CZ
    entangling gates.

    :param num_qubits: int: number of qubits
    :param num_layers: int: number of variational layers

    :return: QuantumCircuit: the VQC
    :return: List[Parameter]: input parameter
    :return: List[Parameter]: variational/weight parameter
    """
    qc = QuantumCircuit(num_qubits)
    input_params = [ Parameter(f"input{q}") for q in range(num_qubits) ]
    weight_params = []

    for l in range(num_layers):

        # input + variational gates
        for q in range(num_qubits):
            y_param = Parameter(f"y{l}_{q}")
            z_param = Parameter(f"z{l}_{q}")
            weight_params.extend([y_param, z_param])

            qc.rx(input_params[q], q)
            qc.ry(y_param, q)
            qc.rz(z_param, q)

        # entangling gates
        for q in range(num_qubits-1):
            qc.cz(q, (q+1) % num_qubits)

    return qc, input_params, weight_params

class VQCModel(BaseFeaturesExtractor):
    """
    :param observation_space: gym Space
    :param num_layers: int: Number of layers
    """
    def __init__(self, observation_space: spaces.Box, num_layers: int = 5):
        num_qubits = observation_space.shape[0]
        super().__init__(observation_space, 2**num_qubits)

        qc, input_params, weight_params = create_VQC(num_qubits, num_layers)
        qnn = SamplerQNN(
                circuit=qc,
                input_params=input_params,
                weight_params=weight_params,
        )
        self.model = TorchConnector(qnn)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Expecting observations in [0, 1] -> scale them to [0, pi]
        observations *= torch.pi

        return self.model(observations)
