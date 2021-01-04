from DQN_Agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import gym

list_of_gyms = [
	'Acrobot-v1',
	'CartPole-v1',
	'MountainCar-v0',
	'LunarLander-v2'
]

GYM_NAME = list_of_gyms[0]
env = gym.make(GYM_NAME)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

DQN_Agent = DQN_Agent.loadmodel()

filename = 'dqn_weights_' + GYM_NAME + '.h5f'
scores = []
epsilons = []	

for trial in range(num_trials):
	done = False
	score = 0
	observation = env.reset()
	while not done:
		env.render()

		action = DQN_Agent.choose_action(observation)

		next_observation, reward, done, info = env.step(action)
	
		score += reward

		observation = next_observation

	epsilons.append(DQN_Agent.epsilon)
	scores.append(score)

