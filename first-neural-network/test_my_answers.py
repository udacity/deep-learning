import os
import sys
import numpy as np
import pandas as pd
from my_answers import NeuralNetwork
from my_answers import iterations, learning_rate, hidden_nodes, output_nodes
import unittest


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


class FirstNNTestMethods(unittest.TestCase):

    ##########
    # Unit tests for data loading
    ##########

    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')

    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))

    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328],
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

    def test_hyperparameters(self):
        bounds = [[50, 15000], [0.05, 5], [5, 100], [1, 1]]
        actual = [iterations, learning_rate, hidden_nodes, output_nodes]
        for ii in range(4):
            self.assertTrue(bounds[ii][0] <= actual[ii] <= bounds[ii][1])

    def test_results(self):
        # Test results of running network on full data
        self.assertTrue(sum(losses['train'][-20:]) / 20.0 < 0.09)
        self.assertTrue(sum(losses['validation'][-20:]) / 20.0 < 0.18)


if __name__ == '__main__':
    data_path = 'Bike-Sharing-Dataset/hour.csv'

    rides = pd.read_csv(data_path)

    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)

    quant_features = ['casual', 'registered', 'cnt',
                      'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean) / std

    # Save data for approximately the last 21 days
    test_data = data[-21 * 24:]

    # Now remove the test data from the data set
    data = data[:-21 * 24]

    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    # Hold out the last 60 days or so of the remaining data as a validation set
    train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
    val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]

    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    losses = {'train': [], 'validation': []}
    for ii in range(iterations):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

        network.train(X, y)

        # Printing out the training progress
        train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

    # Test inputs to test network
    inputs = np.array([[0.5, -0.2, 0.1]])
    targets = np.array([[0.4]])
    test_w_i_h = np.array([[0.1, -0.2],
                           [0.4, 0.5],
                           [-0.3, 0.2]])
    test_w_h_o = np.array([[0.3],
                           [-0.1]])

    #Run the tests
    suite = unittest.TestLoader().loadTestsFromModule(FirstNNTestMethods())
    unittest.TextTestRunner().run(suite)
