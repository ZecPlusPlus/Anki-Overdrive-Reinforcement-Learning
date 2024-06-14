from gymnasium import ObservationWrapper
import numpy as np

class ScalingObservationWrapper(ObservationWrapper):
    def observation(self, observation):
        observation[0] = (observation[0] / 800) * np.pi
        observation[1] = (observation[1] / 47) * np.pi
        observation[2] = (observation[2] / 36) * np.pi
        observation[3] = ((observation[3] + 70) / 140) * np.pi
        return observation
