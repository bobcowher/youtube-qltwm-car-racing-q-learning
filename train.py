from agent import Agent
import gymnasium as gym

env = gym.make("CarRacing-v3", continuous=False, render_mode="human")

agent = Agent(env=env)

agent.train()


