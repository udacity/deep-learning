from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

##################################################
##################################################

import numpy as np
import tensorflow as tf

from .utils import scope_variables_mapping


class Actor:
    def __init__(self, task, critic, scope_name='actor', learning_rate=0.001):
        self.input = tf.placeholder(tf.float32, (None, task.num_states), name='actor/states')
        self.is_training = tf.placeholder(tf.bool, name='actor/is_training')

        self.target = self.create_model(self.input, task, scope_name + '_target')
        self.current = self.create_model(self.input, task, scope_name + '_current', self.is_training)

        self.q_gradients = tf.placeholder(tf.float32, (None, task.num_actions))

        critic = critic.create_model(self.input, self.current, task, critic.scope + '_current', reuse=True)
        loss = tf.reduce_mean(-critic)
        tf.losses.add_loss(loss)

        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = optimizer.minimize(loss, var_list=tf.trainable_variables(scope_name + '_current'))

        self.tau = tf.placeholder(tf.float32)
        self.assignments = [tf.assign(t, c * self.tau + (1-self.tau) * t)
                            for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.init = [tf.assign(t, c)
                     for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.session = None

    def initialize(self):
        self.session.run(self.init)

    def create_model(self, inputs, task, scope_name, training=False):
        g = 0.001
        eps = 1
        with tf.variable_scope(scope_name):
            dense = tf.layers.dense(inputs, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=eps)
            # dense = tf.layers.batch_normalization(dense, training=training)

            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=eps)
            # dense = tf.layers.batch_normalization(dense, training=training)

            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=eps)
            # dense = tf.layers.batch_normalization(dense, training=training)

            dense = tf.layers.dense(dense, task.num_actions,
                                    activation=tf.nn.sigmoid,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-g, maxval=g),
                                    bias_initializer=tf.random_uniform_initializer(minval=-g, maxval=g))

            action_min = np.array([task.action_low] * task.num_actions)
            action_max = np.array([task.action_high] * task.num_actions)
            action_range = action_max - action_min

            result = dense * action_range + action_min

        return result

    def set_session(self, session):
        self.session = session

    def get_action(self, state):
        return self.session.run(
            self.current,
            feed_dict={
                self.input: state,
                self.is_training: False})

    def get_target_action(self, state):
        return self.session.run(self.target, feed_dict={self.input: state})

    def learn(self, state):
        self.session.run(
            self.optimizer,
            feed_dict={
                self.input: state,
                self.is_training: True})

    def update_target(self, tau):
        self.session.run(self.assignments, feed_dict={self.tau: tau})
