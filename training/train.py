import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.mario_env import make_mario_env

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config("config/ppo_config.yaml")
    env = make_mario_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./models/ppo/',
        name_prefix='mario_ppo_checkpoint'
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        tensorboard_log="./models/logs/tensorboard/"
    )

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=checkpoint_callback
    )

    model.save("./models/ppo/mario_ppo_v1")

if __name__ == "__main__":
    main()
