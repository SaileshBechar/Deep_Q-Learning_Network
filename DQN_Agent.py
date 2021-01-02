'''
Script to create a Deep Q-Network using Keras to solve problems in Open AI Gym with a discrete action space vector (no convolutions)

Sailesh Bechar 
Winter 2021
'''
import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

def build_dqn(learning_rate, num_actions, input_dimensions, layer1_dims, layer2_dims):
	model = Sequential()
	model.add(Dense(layer1_dims, input_shape=(input_dimensions,), activation='relu')) # Input dimensions should correspond to observation space dims
	model.add(Dense(layer2_dims, activation='relu'))
	model.add(Dense(num_actions, activation='linear')) # Output dimensions should be a 1-1 mapping to possible actions in environment

	model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

	return model

class ReplayBuffer:
	def __init__ (self, max_length):
		self.replay_buffer = deque(maxlen=max_length)
		self.memory_counter = 0

	def store_experience(self, state, action, reward, next_state, done):
		self.replay_buffer.append([state, action, reward, next_state, done])
		self.memory_counter += 1

	def sample_buffer(self, batch_size):
		return random.choices(self.replay_buffer, k=batch_size)

	def get_index(self, index):
		return self.replay_buffer[index]

class DQNAgent:
	def __init__(self, learning_rate, num_actions, input_dimensions, discount_factor, batch_size, epsilon,
					epsilon_decrement=0.9999, epsilon_min=0.01, replay_buffer_size=2000, file_name='dqn_model.h5'):

		self.num_actions = num_actions
		self.discount_factor = discount_factor
		self.batch_size = batch_size
		self.epsilon = epsilon
		self.epsilon_decrement = epsilon_decrement
		self.epsilon_min = epsilon_min
		self.file_name = file_name

		self.Replay_Buffer = ReplayBuffer(replay_buffer_size)
		self.dqn = build_dqn(learning_rate, num_actions, input_dimensions, 64, 64)
		self.target_network = build_dqn(learning_rate, num_actions, input_dimensions, 64, 64) # Trick to improve convergence by allowing the network to now approximate a static destination

	def remember_experience(self, state, action, reward, next_state, done):
		self.Replay_Buffer.store_experience(state, action, reward, next_state, done)

	def choose_action(self, state):
		self.epsilon *= self.epsilon_decrement
		self.epsilon = max(self.epsilon_min, self.epsilon)

		if np.random.random() < self.epsilon or self.Replay_Buffer.memory_counter < self.batch_size:
			action = random.randrange(self.num_actions)
		else:
			possible_actions = self.dqn.predict(state)
			action = np.argmax(possible_actions[0])

		return action

	def train(self):
		# print(self.Replay_Buffer.memory_counter)
		if self.Replay_Buffer.memory_counter > self.batch_size:
			samples = self.Replay_Buffer.sample_buffer(self.batch_size)
			
			for sample in samples: # Find best possible action from each experience in batch
				state, action, reward, next_state, done = sample
				target = self.target_network.predict(state)
				if done:
					target[0][action] = reward
				else:
					q_next = max(self.target_network.predict(next_state)[0])
					target[0][action] = reward + q_next * self.discount_factor # Find best action based on max potential reward
				self.dqn.fit(state, target, epochs=1, verbose=0)

	def train_target_network(self):
		# Hyper parameter, adjust weights of target network by weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		self.target_network.set_weights(self.dqn.get_weights()) 

	def save_model(self):
		self.dqn.save(self.file_name)		

	def load_model(self):
		self.dqn = load_model(self.file_name)

