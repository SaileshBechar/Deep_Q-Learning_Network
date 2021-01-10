from DQN_Agent import DQNAgent
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gym
import time

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
scores = []

folder = GYM_NAME + '/' + 'Acrobot-v1_20210109-140145' + '/'
filename = 'dqn_weights_' + GYM_NAME + '.h5'

epsilon =  0 # Hyper Parameter , try 0.5
learning_rate = 0.0005 # Hyper Parameter , try {0.001, 0.005}
gamma = 0.9 # Hyper Parameter, try 0.99
batch_size = 32 # Hyper Parameter, try 64
num_trials = 10

DQN_Agent = DQNAgent(learning_rate=learning_rate, num_actions=num_actions, input_dimensions=num_states,
				discount_factor=gamma, epsilon=epsilon, batch_size=batch_size)

DQN_Agent.load_model(folder + filename)

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


	scores.append(score)
	
	time.sleep(0.5)

	avg_score = np.mean(scores[max(0, trial-100):(trial+1)])
	print('Trial number {}":'.format(trial), 'score: %.2f' % score,
	' average score %.2f' % avg_score)


