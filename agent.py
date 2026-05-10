import gymnasium as gym
import cv2
import torch
import random
from buffer import ReplayBuffer
from models.q_model import QModel

class Agent:

    def __init__(self, env: gym.Env,
                       max_buffer_size: int = 20000,
                       target_update_interval: int = 10000) -> None:
        self.env = env
        self.epsilon = 1

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        obs, _ = self.env.reset()
        obs = self.process_observation(obs)

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=obs.shape,
            input_device=self.device,
            output_device=self.device
        )

        print(f"Initializing agent on device {self.device}")

        self.q_model = QModel(
            action_dim=self.env.action_space.n,
            input_shape=obs.shape,
        ).to(self.device)


    def process_observation(self, obs):
        obs = cv2.resize(obs, (96,96), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1)
        return obs

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return random.choices([0, 1, 2, 3, 4], weights=[0.05, 0.20, 0.20, 0.50, 0.05])[0]
        else: 
            # TODO: Model selects action
            pass
        


    def train(self, episodes=1, batch_size=32):

        for episode in range(episodes):
            obs, _ = self.env.reset()

            done = False
            episode_reward = 0.0
            episode_loss = 0.0
            episode_steps = 0

            while not done:

                action = self.select_action(obs)

                next_obs, reward, term, trunc, _ = self.env.step(action) 
                done = term or trunc
