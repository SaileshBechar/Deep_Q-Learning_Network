from DQN_Agent import DQNAgent
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gym
import time

def plot_learning(x_data, score_data, avg_data, epsilon_data):
	fig, axs = plt.subplots(2, sharex=True)
	fig.suptitle('DQN results for ' + GYM_NAME)

	axs[0].scatter(x_data, score_data, c='tab:orange', label='Score')
	axs[0].plot(x_data, avg_data, label='Avg of last 100 scores')
	axs[0].legend()
	axs[0].set_ylabel("Score")
	axs[1].plot(x_data, epsilon_data)
	axs[1].set_ylabel("Eplison")

	axs[1].set_xlabel("Trial Number")

	for ax in fig.get_axes():
		ax.label_outer()

	plt.savefig(folder + filename + '_' + str(len(x_data)) + '.png')

list_of_gyms = [
	'Acrobot-v1',
	'CartPole-v1',
	'MountainCar-v0',
	'LunarLander-v2'
]

GYM_NAME = list_of_gyms[2]
env = gym.make(GYM_NAME)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

epsilon =  0.95 # Hyper Parameter , try 0.5
learning_rate = 0.0005 # Hyper Parameter , try {0.001, 0.005}
gamma = 0.9 # Hyper Parameter, try 0.99
batch_size = 64 # Hyper Parameter, try 64
num_trials = 500

DQN_Agent = DQNAgent(learning_rate=learning_rate, num_actions=num_actions, input_dimensions=num_states,
				discount_factor=gamma, epsilon=epsilon, batch_size=batch_size)

folder = GYM_NAME + '_' + time.strftime("%Y%m%d-%H%M%S") + '/'
Path(folder).mkdir(parents=True, exist_ok=True)
filename = 'dqn_weights_' + GYM_NAME
scores = []
avg_scores = []
epsilons = []	
start_time = time.time()

for trial in range(num_trials):
	done = False
	score = 0
	observation = env.reset()
	while not done:
		action = DQN_Agent.choose_action(observation)
		
		next_observation, reward, done, info = env.step(action)
	
		score += reward

		DQN_Agent.remember_experience(observation, action, reward, next_observation, done)
		DQN_Agent.train()
		DQN_Agent.train_target_network()

		observation = next_observation

	epsilons.append(DQN_Agent.epsilon)
	scores.append(score)

	avg_score = np.mean(scores[max(0, trial-100):(trial+1)])
	print('Trial number {}":'.format(trial), 'score: %.2f' % score,
	' average score %.2f' % avg_score)
	avg_scores.append(avg_score)

	if trial % 10 == 0:
		DQN_Agent.save_model(folder + filename + '.h5')
		x_data = [i+1 for i in range(trial + 1)]
		plot_learning(x_data, scores, avg_scores, epsilons)

print("--- Trained in %s seconds ---" % (time.time() - start_time))