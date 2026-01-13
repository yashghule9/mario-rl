import cv2
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, RewardWrapper

class ResizeFrame(ObservationWrapper):
    def __init__(self, env, shape=(84,84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs, -1)


class RewardShaper(RewardWrapper):
    def reward(self, reward):
        # shaping rewards for better learning
        reward = reward / 10.0  
        return reward
