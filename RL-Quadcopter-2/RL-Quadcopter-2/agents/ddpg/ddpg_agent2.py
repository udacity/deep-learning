#from pathlib import Path
import numpy as np
import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense

from agents.agent import Agent

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
        
    def reset(self):
        self.episode_score = 0.0
        #self.episode_ticks = 0
        self.noise.reset()
        return self.task.reset()

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        #self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

########


class DeepDPGAgent(BaseAgent):
    batch_size = 64
    tau = 0.001
    gamma = 0.99
    learning_rate = 0.0001

    """Implement Deep DPG control agent

    From paper by Lillicrap, Timothy P. "Continuous Control with Deep Reinforcement Learning."
    https://arxiv.org/pdf/1509.02971.pdf

    """
    def __init__(self, task, replay_buffer_size=100000, batch_size=None):
        """Initialize policy and other agent parameters.

        Should be able to access the following (OpenAI Gym spaces):
            task.observation_space  # i.e. state space
            task.action_space
        """
        super().__init__(task)

        # Create actor and critic
        self.critic = Critic(task, learning_rate=DeepDPGAgent.learning_rate * 100)
        self.actor = Actor(task, self.critic, learning_rate = DeepDPGAgent.learning_rate)

        #self.noise = OUNoise2(
        #    task.num_actions,
        #    theta=0.15,
        #    sigma=0.2)
        self.noise = OUNoise(
            task.num_actions,
            theta=0.15,
            sigma=25)

        # Create critic NN

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.actor.set_session(self.session)
        self.critic.set_session(self.session)

        self.actor.initialize()
        self.critic.initialize()

        # writer = tf.summary.FileWriter('graph', self.session.graph)

        self.prev_state = None

        self.batch_size = batch_size or DeepDPGAgent.batch_size
        self.tau = DeepDPGAgent.tau
        self.gamma = DeepDPGAgent.gamma

        self.best_score = None
        self.episode_score = 0.0
        self.episode_ticks = 0

        self.episode = 1

        self.saver = tf.train.Saver()

        self.load_task_agent()

    def reset_episode(self):
        self.episode_score = 0.0
        self.episode_ticks = 0
        self.noise.reset()
        return self.task.reset()

    def act(self, state):
        action = self.actor.get_action(np.expand_dims(state, axis=0))[0]

        noise = self.noise.sample()
        for i in range(self.task.num_actions):
            noise[i] = min(self.task.action_high, max(self.task.action_low, noise[i]))

        action += noise

        return action

    def act_target(self, state):
        action = self.actor.get_target_action(np.expand_dims(state, axis=0))[0]
        return action

    def step(self, action, reward, next_state, done):
        self.episode_ticks += 1
        self.episode_score += reward

        if self.prev_state is not None:
            self.replay_buffer.add([self.prev_state, action, reward, next_state, done])

        if len(self.replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)

            prev_states = np.array([t[0] for t in batch])
            prev_actions = np.array([t[1] for t in batch])
            rewards = np.expand_dims(np.array([t[2] for t in batch]), axis=1)
            states = np.array([t[3] for t in batch])

            y = rewards + self.gamma * self.critic.get_target_value(states, self.actor.get_target_action(states))
            self.critic.learn(prev_states, prev_actions, y)
            self.actor.learn(prev_states)

            self.critic.update_target(self.tau)
            self.actor.update_target(self.tau)

        if done:
            if self.best_score is not None:
                self.best_score = max(self.best_score, self.episode_score)
            else:
                self.best_score = self.episode_score
            self.episode += 1
            self.save_task_agent()

        self.prev_state = next_state if not done else None

    def show_episode_stats(self):
        print("Deep DPG episode stats: t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
            self.episode_ticks, self.episode_score, self.best_score, 0))# self.noise_scale))  # [debug]

    @property
    def noise_scale(self):
        return 0

    def load(self, path):
        self.saver.restore(self.session, path)

    def save(self, path):
        self.saver.save(self.session, path)

