#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:48:08 2022

@author: johnsalvaris
"""

# Import packages
import sklearn
import string
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import learning_curve


class sklearn_SVR():
    def __init__(self, kernel='rbf', gamma='scale', epsilon='0.1', C=1.0, tolerance=0.001, degree=3, coef0=0.0, shrinking=True, verbose=0, shuffle_order=False, epochs=100):
        self.model_type = 'SVR'
        self.kernel = kernel
        self.gamma = gamma  # 'scale' , 'auto'
        self.epsilon = epsilon  # epsilon tube for no penalty to training loss function
        self.verbose = verbose
        self.tol = tolerance  # tolerance for stopping criteria
        self.C = C  # Regularisation Parameter
        self.degree = degree
        self.coef0 = coef0
        self.shuffle = shuffle_order
        self.shrinking = shrinking
        self.epochs = epochs

        if self.verbose == 0:
            self.verbose = False
        elif self.verbose == 1 or self.verbose == 2:
            self.verbose = True
        else:
            raise Exception("Invalid verbose. Redefine as 0 (off), 1 or 2 (on)")


    def add_sets(self, model_parameters):
        self.X_train = model_parameters.X_train
        self.y_train = model_parameters.y_train
        self.unscaled_y_train = model_parameters.unscaled_y_train.to_numpy()
        self.X_test = model_parameters.X_test
        self.y_test = model_parameters.y_test
        self.unscaled_y_test = model_parameters.unscaled_y_test.to_numpy()
        self.train_dates = model_parameters.train_dates.to_numpy()
        self.test_dates = model_parameters.test_dates.to_numpy()

        self.train_output_swl = model_parameters.train_output_swl.to_numpy()
        self.test_output_swl = model_parameters.test_output_swl.to_numpy()
        self.train_output_current_swl = model_parameters.train_output_current_swl.to_numpy()
        self.test_output_current_swl = model_parameters.test_output_current_swl.to_numpy()

        self.gwl_input = model_parameters.gwl_input
        self.gwl_output = model_parameters.gwl_output


    def create_model(self):
        self.model = SVR(kernel=self.kernel, gamma=self.gamma, epsilon=self.epsilon, C=self.C, tol=self.tol, degree=self.degree, coef0=self.coef0, verbose=self.verbose)


    def train_model(self, X, y, scoring_code='neg_mean_squared_error'):
        self.model.fit(X, y)  # Possible improvement --> Add sample weights to force more empahsis on certain features?
        self.scoring_code = scoring_code

        self.epoch_list, self.train_curve_scores, self.test_curve_scores, self.fit_times, self.score_times = learning_curve(self.model, X, y, return_times=True, scoring=scoring_code, train_sizes=np.linspace(1 / self.epochs, 1, self.epochs))
        if self.epochs > self.epoch_list.shape[0]:
            print("Please ignore above error")
        if self.scoring_code[:3] == 'neg':
            self.test_curve_scores = -self.test_curve_scores
            self.train_curve_scores = -self.train_curve_scores
            self.scoring_code = self.scoring_code[4:]
            self.scoring_code = self.scoring_code.replace("_", " ")
            self.scoring_code = string.capwords(self.scoring_code)
        self.train_curve_scores_mean = np.mean(self.train_curve_scores, axis=1)
        self.train_curve_scores_std = np.std(self.train_curve_scores, axis=1)
        self.test_curve_scores_mean = np.mean(self.test_curve_scores, axis=1)
        self.test_curve_scores_std = np.std(self.test_curve_scores, axis=1)
        self.fit_times_mean = np.mean(self.fit_times, axis=1)
        self.fit_times_std = np.std(self.fit_times, axis=1)


    def predict(self, X, scaler, dataset='test'):
        if dataset == 'train':
            self.y_hat_train = self.model.predict(X).reshape(X.shape[0], 1)
            self.unscaled_y_hat_train = scaler.inverse_transform(self.y_hat_train)
            if self.gwl_output in ['delta', 'average_delta']:
                self.y_hat_level_train = np.array(list(map(lambda current, change: current + change, self.train_output_current_swl, self.unscaled_y_hat_train)))
        elif dataset == 'test':
            self.y_hat_test = self.model.predict(X).reshape(X.shape[0], 1)
            self.unscaled_y_hat_test = scaler.inverse_transform(self.y_hat_test)
            if self.gwl_output in ['delta', 'average_delta']:
                self.y_hat_level_test = np.array(list(map(lambda current, change: current + change, self.test_output_current_swl, self.unscaled_y_hat_test)))
        else:
            raise Exception("Specify if dataset is 'train' or 'test'")
