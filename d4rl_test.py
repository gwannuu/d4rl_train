import d4rl
import gym

env = gym.make("antmaze-umaze-v0")
data_dict = d4rl.qlearning_dataset(env)
print(data_dict)
