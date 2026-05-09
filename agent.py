import gymnasium as gym

class Agent:

    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def train(self, episodes=1, batch_size=32):

        for episode in range(episodes):
            obs, _ = self.env.reset()

            done = False
            episode_reward = 0.0
            episode_loss = 0.0
            episode_steps = 0

            while not done:

                action = self.env.action_space.sample()

                next_obs, reward, term, trunc, _ = self.env.step(action) 
                done = term or trunc
