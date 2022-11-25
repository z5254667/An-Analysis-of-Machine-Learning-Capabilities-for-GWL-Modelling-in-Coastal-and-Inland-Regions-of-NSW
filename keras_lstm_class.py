#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:39:13 2022

@author: johnsalvaris
"""

# Import packages
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
# from tensorflow.keras.metrics import RootMeanSquaredError as RMSE


class keras_LSTM():
    def __init__(self, optimiser='adam', loss='mse', metric='accuracy', epochs=200, keras_validation_split=0.2, shuffle_order=False, verbose=0):
        self.model_type = 'LSTM'
        self.optimiser = optimiser
        self.loss = loss  # mae, mse, rmse
        self.metric = metric
        self.epochs = epochs
        self.verbose = verbose
        self.val_split = keras_validation_split
        self.shuffle_order = shuffle_order


    def add_sets(self, model_parameters):
        self.X_train = model_parameters.X_train
        self.y_train = model_parameters.y_train
        self.train_dates = model_parameters.train_dates.to_numpy()
        self.test_dates = model_parameters.test_dates.to_numpy()
        self.unscaled_y_train = model_parameters.unscaled_y_train.to_numpy()
        self.X_test = model_parameters.X_test
        self.y_test = model_parameters.y_test
        self.unscaled_y_test = model_parameters.unscaled_y_test.to_numpy()

        self.train_output_swl = model_parameters.train_output_swl.to_numpy()
        self.test_output_swl = model_parameters.test_output_swl.to_numpy()
        self.train_output_current_swl = model_parameters.train_output_current_swl.to_numpy()
        self.test_output_current_swl = model_parameters.test_output_current_swl.to_numpy()

        self.gwl_input = model_parameters.gwl_input
        self.gwl_output = model_parameters.gwl_output


    def create_network(self, input_shape, num_lstm_cells=64, num_fc_neurons=8, lstm_dropout_rate=0, lstm_recurrent_dropout_rate=0, lstm_layers=1, fch_layers=1, num_outputs=1):
        self.full_input_shape = input_shape
        self.input_shape = input_shape[1:]  # (lags, features)
        self.num_outputs = num_outputs  # number of outputs
        self.lstm_cells = num_lstm_cells
        self.fc_neurons = num_fc_neurons
        self.lstm_dropout = lstm_dropout_rate
        self.lstm_recurrent_dropout = lstm_recurrent_dropout_rate
        self.lstm_layers = lstm_layers
        self.fch_layers = fch_layers

        self.model = Sequential()

        # Input layer
        self.model.add(InputLayer(input_shape=self.input_shape))

        # LSTM Layers
        l = 1
        while l <= self.lstm_layers - 1:
            self.model.add(LSTM(self.lstm_cells, dropout=self.lstm_dropout, recurrent_dropout=self.lstm_recurrent_dropout, return_sequences=True))
            l += 1
        self.model.add(LSTM(self.lstm_cells, dropout=self.lstm_dropout, recurrent_dropout=self.lstm_recurrent_dropout))

        # Fully Connected Hidden Layers
        f = 1
        while f <= self.fch_layers:
            self.model.add(Dense(self.fc_neurons, activation='relu'))
            f += 1

        # Fully Connected Output Layer
        self.model.add(Dense(self.num_outputs))  # linear Activtion

        # compile
        self.model.compile(optimizer=self.optimiser, loss=self.loss, metrics=[self.metric])


    def train_model(self, X, y):
        self.history = self.model.fit(X, y, epochs=self.epochs, verbose=self.verbose, validation_split=self.val_split, shuffle=self.shuffle_order)
        self.train_loss, self.train_metric = self.model.evaluate(X, y, verbose=self.verbose)


    def predict(self, X, scaler, dataset='test'):
        if dataset == 'train':
            self.y_hat_train = self.model.predict(X, verbose=self.verbose)
            self.unscaled_y_hat_train = scaler.inverse_transform(self.y_hat_train)
            if self.gwl_output in ['delta', 'average_delta']:
                self.y_hat_level_train = np.array(list(map(lambda current, change: current + change, self.train_output_current_swl, self.unscaled_y_hat_train)))
        elif dataset == 'test':
            self.y_hat_test = self.model.predict(X, verbose=self.verbose)
            self.unscaled_y_hat_test = scaler.inverse_transform(self.y_hat_test)
            if self.gwl_output in ['delta', 'average_delta']:
                self.y_hat_level_test = np.array(list(map(lambda current, change: current + change, self.test_output_current_swl, self.unscaled_y_hat_test)))
        else:
            raise Exception("Specify if dataset is 'train' or 'test'")
