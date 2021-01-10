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
		self.replay_buffer.append((np.array(state), action, reward, np.array(next_state), done))
		self.memory_counter += 1

	def sample_buffer(self, batch_size):
		sample_size = min(len(self.replay_buffer), batch_size)
		samples = random.choices(self.replay_buffer, k=sample_size)
		return map(list, zip(*samples))

	def get_index(self, index):
		return self.replay_buffer[index]

class DQNAgent:
	def __init__(self, learning_rate, num_actions, input_dimensions, discount_factor, batch_size, epsilon,
					epsilon_decrement=0.99, epsilon_min=0.01, replay_buffer_size=1_000_000):

		self.num_actions = num_actions
		self.discount_factor = discount_factor
		self.batch_size = batch_size
		self.epsilon = epsilon
		self.epsilon_decrement = epsilon_decrement
		self.epsilon_min = epsilon_min
		self.learning_step = 0
		self.q_vals = []
		self.Replay_Buffer = ReplayBuffer(replay_buffer_size)
		self.dqn = build_dqn(learning_rate, num_actions, input_dimensions, 512, 512)
		self.target_network = build_dqn(learning_rate, num_actions, input_dimensions, 512, 512) # Trick to improve convergence by allowing the network to now approximate a static destination

	def remember_experience(self, state, action, reward, next_state, done):
		self.Replay_Buffer.store_experience(state, action, reward, next_state, done)

	def choose_action(self, state):
		self.epsilon = max(self.epsilon_min, self.epsilon_decrement * self.epsilon)
		state = np.array([state], dtype=np.float32)
		possible_actions = self.dqn.predict(state)
		self.q_vals.append(np.max(possible_actions[0]))

		if np.random.random() < self.epsilon:
			action = random.randrange(self.num_actions)
		else:
			action = np.argmax(possible_actions[0])

		return action

	def train(self):
		if self.Replay_Buffer.memory_counter > self.batch_size:
			states, actions, rewards, next_states, dones = self.Replay_Buffer.sample_buffer(self.batch_size) # Retrieves a batch_size amount of each val
			states = np.array(states)
			next_states = np.array(next_states)

			q_eval_states = self.dqn.predict(states)
			q_next_states = self.target_network.predict(next_states)

			q_next_states[dones] = np.zeros([self.num_actions])
			q_target = q_eval_states[:]
			indices = np.arange(self.batch_size)
			q_target[indices, actions] = rewards + self.discount_factor * np.max(q_next_states, axis=1) 

			self.dqn.fit(states, q_target, verbose=0)

	def train_target_network(self):
		self.target_network.set_weights(self.dqn.get_weights()) 

	def save_model(self, file_name):
		self.dqn.save(file_name)		

	def load_model(self, file_name):
		self.dqn = load_model(file_name)
		self.target_network = load_model(file_name)

