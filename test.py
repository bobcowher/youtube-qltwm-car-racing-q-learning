from agent import Agent
import gymnasium as gym

env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
# env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

agent = Agent(env=env, max_buffer_size=10)

agent.load()

agent.test(episodes=1)


