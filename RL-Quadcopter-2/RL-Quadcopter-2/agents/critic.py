#from keras import layers, models, optimizers, losses
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras import backend as K

#from keras.utils.np_utils import to_categorical

import tensorflow as tf

from .utils import scope_variables_mapping

class Critic:
    """Critic (Value) Model."""
    
    def __init__(self, input_states, input_actions, task,  scope_name='critic', training=False, reuse=False):
    ## def __init__(self, task, scope_name, training=False, reuse=False):
        """Initialize parameters and build model.

        Params
        ======
            input_states (int): Dimension of each state
            input_actions (int): Dimension of each action
            is_training
            learning_rate
            gamma
            tau
            target
            current
        """
        self.scope = scope_name
        self.learning_rate = 0.001
        self.input_states=input_states
        self.input_actions=input_actions

        #self.input_states = tf.placeholder(
        #    tf.float32,
        #    (None, self.input_states),
        #    name='critic/states')
        
        self.input_states = tf.placeholder(
            tf.float32,
            (None, task.num_states),
            name='critic/states')
        
        #self.input_actions = tf.placeholder(
        #    tf.float32,
        #    (None, self.input_actions),
        #    name='critic/actions')
        
        self.input_actions = tf.placeholder(
            tf.float32,
            (None, task.num_actions),
            name='critic/actions')
        
        self.is_training = tf.placeholder(tf.bool, name='critic/is_training')

        ## print("DDPG scope_name = {} \n".format(scope_name + '_target'), end="")  # [debug]
        self.target = self.build_model(self.input_states, self.input_actions, task, scope_name + '_target')
        self.current = self.build_model(self.input_states, self.input_actions, task, scope_name + '_current', training=self.is_training)

        self.y = tf.placeholder(tf.float32, (None, 1), name='critic/y')
        loss = tf.losses.mean_squared_error(self.y, self.current)
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        self.tau = tf.placeholder(tf.float32, name='critic/tau')
        self.assignments = [tf.assign(t, c * self.tau + (1-self.tau) * t)
                            for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.init = [tf.assign(t, c)
                     for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.session = None

        # Initialize any other variables here

        # self.build_model(self, self.input_states, self.input_actions, task, scope_name) 
        
    def initialize(self):
        self.session.run(self.init)
        
    def build_model(self, input_states, input_actions, task, scope_name, training=False, reuse=False): 
        ## reuse=tf.AUTO_REUSE): # with tf.Graph().as_default():
        with tf.variable_scope(scope_name, reuse=reuse):
            g = 0.0001
            # 2 layers of states
            dense_s1 = tf.layers.dense(input_states, 64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())#,
                                      #reuse=False)
            # dense_s = tf.layers.dropout(dense_s, 0.5, training=training)
            # dense_s = tf.nn.l2_normalize(dense_s, epsilon=0.01)
            # dense_s = tf.layers.batch_normalization(dense_s, training=training)

            dense_s = tf.layers.dense(dense_s1, 64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())#,
                                      #reuse=True)
            # dense_s = tf.layers.dropout(dense_s, 0.5, training=training)
            # dense_s = tf.nn.l2_normalize(dense_s, epsilon=0.01)
            # dense_s = tf.layers.batch_normalization(dense_s, training=training)
            # dense_s = tf.reset_default_graph()

            # One layer of actions
            dense_a = tf.layers.dense(input_actions, 64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())#,
                                      #reuse=False)
            # dense_a = tf.layers.dropout(dense_a, 0.5, training=training)
            # dense_a = tf.nn.l2_normalize(dense_a, epsilon=0.01)
            # dense_a = tf.layers.batch_normalization(dense_a, training=training)

            # Merge together
            dense = tf.concat([dense_s, dense_a], axis=1)

            # Decision layers
            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=0.01)
            # dense = tf.layers.batch_normalization(dense, training=training)

            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            # dense = tf.layers.dropout(dense, 0.5, training=training)
            # dense = tf.nn.l2_normalize(dense, epsilon=0.01)
            # dense = tf.layers.batch_normalization(dense, training=training)

            # Output layer
            dense = tf.layers.dense(dense, 1,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-g, maxval=g),
                                    bias_initializer=tf.random_uniform_initializer(minval=-g, maxval=g))
            result = dense

        return result
    
    def set_session(self, session):
        self.session = session

    def get_value(self, state, action):
        return self.session.run(
            self.current,
            feed_dict={self.input_states: state, self.input_actions: action, self.is_training: False})

    def get_target_value(self, state, action):
        return self.session.run(
            self.target,
            feed_dict={self.input_states: state, self.input_actions: action, self.is_training: False})

    def learn(self, states, actions, targets):
        self.session.run(
            self.optimizer,
            feed_dict={
                self.input_states: states,
                self.input_actions: actions,
                self.y: targets,
                self.is_training: True})

    def update_target(self, tau):
        self.session.run(self.assignments, feed_dict={self.tau: tau})
    
    
        """
    def build_model2(self):
        
        ## x = tf.placeholder(tf.float32, [None, 1024])
        ## y = keras.layers.Dense(512, activation='relu')(x)
        # Define input layers
        ## states = layers.Input(shape=(self.state_size,), name='states')
        ## actions = layers.Input(shape=(self.action_size,), name='actions')
        
        # Define input layers
        states_tf = tf.placeholder(tf.float32, [None, self.state_size])
        actions_tf = tf.placeholder(tf.float32, [None, self.action_size])

        # Add hidden layer(s) for state with Keras
        h_states = layers.Dense(512, activation='relu')(states_tf)
        h_states = layers.Dense(units=64, activation='relu')(states_tf)
        h_states = layers.Dense(units=128, activation='relu')(states_tf)
        h_states = layers.Dense(units=1, activation='sigmoid')(states_tf)
        h_states = layers.Dropout(0.5)(h_states)
        #h_states = layers.Flatten()

        # Add hidden layer(s) for action with Keras
        h_actions = layers.Dense(512, activation='relu')(actions_tf)
        h_actions = layers.Dense(units=64, activation='relu')(h_actions)
        h_actions = layers.Dense(units=128, activation='relu')(h_states)
        h_actions = layers.Dense(units=1, activation='sigmoid')(h_states)
        h_actions = layers.Dropout(0.5)(h_states)
        #h_actions = layers.Flatten()

        # Combine state and action pathways
        q_values = layers.Add()([h_states, h_actions])
        q_values = layers.Activation('relu')(q_values)

        # Add final output layer to produce action values (Q values)
        ql_values = layers.Dense(units=1, name='q_values')(q_values)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=ql_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(self.learning_rate, self.gamma, self.tau)
        self.model.compile(optimizer=optimizer,
                           loss='mse', #loss='categorical_crossentropy'
                           metrics=['accuracy'])

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(ql_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        
        result = self.get_action_gradients

        return result
    """
