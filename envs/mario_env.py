import gym_super_mario_bros
import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from .wrappers import ResizeFrame, RewardShaper

def make_mario_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", render_mode=None)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env = ResizeFrame(env, shape=(84,84))
    env = RewardShaper(env)

    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    return env
