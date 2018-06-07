#from pathlib import Path
import numpy as np
import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense

# from agents.agent import Agent

from agents.ddpg.ddpg_actor import Actor
from agents.ddpg.ddpg_critic import Critic
from agents.ddpg.OUNoise import OUNoise
from agents.ddpg.Replay_Buffer import ReplayBuffer

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, learning_rate):
		
		self.task = task
		self.state_size = task.state_size
		self.action_size = task.action_size
		self.action_low = task.action_low
		self.action_high = task.action_high
		self.last_state = task.done
		self.learning_rate = learning_rate

		# Actor (Policy) Model
		self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
		self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

		# Critic (Value) Model
		self.critic_local = Critic(self.state_size, self.action_size)
		self.critic_target = Critic(self.state_size, self.action_size)

		# Initialize target model parameters with local model parameters
		self.critic_target.model.set_weights(self.critic_local.model.get_weights())
		self.actor_target.model.set_weights(self.actor_local.model.get_weights())

		# Noise process
		self.exploration_mu = 0
		self.exploration_theta = 0.15
		self.exploration_sigma = 0.2
		self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

		# Replay memory
		self.buffer_size = 100000
		self.batch_size = 64
		self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

		# Algorithm parameters
		self.gamma = 0.99  # discount factor
		self.tau = 0.01  # for soft update of target parameters

		# Score tracker and learning parameters
		# self.best_score = -np.inf
		# self.noise_scale = 0.1
        
    def reset(self):
        self.episode_score = 0.0
        self.episode_ticks = 0
        self.noise.reset()
        return self.task.reset()

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def step(self, action, reward, next_state, done):
        self.episode_ticks += 1
        self.episode_score += reward
		
		# setup through episode's states
		# self.prev_state = last_state

        if self.prev_state is not None:
            # self.replay_buffer.add([self.prev_state, action, reward, next_state, done])
			# Save experience / reward
			self.memory.add(self.last_state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            batch = self.memory.sample(self.batch_size)

			# Learn, if enough samples are available in memory
			if len(self.memory) > self.batch_size:
				experiences = self.memory.sample()
				self.learn(experiences)
		
			#y = rewards + self.gamma * self.critic.get_target_value(states, self.actor.get_target_action(states))
			#self.critic.learn(prev_states, prev_actions, y)
			#self.actor.learn(prev_states)

			#self.critic.update_target(self.tau)
			#self.actor.update_target(self.tau)
		
			# Roll over last state and action
			# self.last_state = next_state
		
		if done:
            if self.best_score is not None:
                self.best_score = max(self.best_score, self.episode_score)
            else:
                self.best_score = self.episode_score
            self.episode += 1
            # self.save_task_agent()

        self.prev_state = next_state if not done else None
			