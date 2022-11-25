#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:44:53 2022

@author: johnsalvaris
"""

# Import packages
import numpy as np


# Root Mean Squared Error
def rmse(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    return np.sqrt(np.sum((actual - predicted)**2) / actual.shape[0])


# Mean Squared Error
def mse(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    return np.sum((actual - predicted)**2) / actual.shape[0]


# Coefficient of Determination
def r_squared(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    rss = np.sum((actual - predicted)**2)
    actual_mean = np.mean(actual)
    tss = np.sum((actual - actual_mean)**2)
    return 1 - rss / tss


# Mean Absolue Percentage Error --> Note calculates decimal value
def mape(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    return np.sum(np.abs((actual - predicted) / actual)) / actual.shape[0]


# Mean Absolute Error
def mae(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    return np.sum(np.abs(predicted - actual)) / actual.shape[0]


# Normalised Nash-Sutcliffe Efficiency
def nnse(actual, predicted):
    nse = r_squared(actual, predicted)  # Note: NSE is cacluated idential to r^2

    return 1 / (2 - nse)


# Pearson's Correlation Coefficient
def r(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    actual_mean = np.mean(actual)
    predicted_mean = np.mean(predicted)

    numerator = np.sum((actual - actual_mean) * (predicted - predicted_mean))
    denominator = np.sqrt(np.sum((actual - actual_mean)**2) * np.sum((predicted - predicted_mean)**2))

    return numerator / denominator


# Index of Agreement
def ia(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    actual_mean = np.mean(actual)
    numerator = np.sum((actual - predicted)**2)
    denominator = np.sum((abs(predicted - actual_mean) + abs(actual - actual_mean))**2)

    return max(min(1 - numerator / denominator, 1), 0)


# Kling-Gupta Efficiency
def kge(actual, predicted):
    cc = r(actual, predicted)

    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    average_actual = np.mean(actual)
    average_predicted = np.mean(predicted)
    std_actual = np.std(actual)
    std_predicted = np.std(predicted)

    return 1 - np.sqrt((cc - 1)**2 + (std_predicted / std_actual - 1)**2 + (average_predicted / average_actual - 1)**2)


# Mean Bias Error
def mbe(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    return 1 / len(actual) * np.sum(predicted - actual)


# Normalised Root Mean Square Error
def nrmse(actual, predicted):
    rmsd = rmse(actual, predicted)

    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])

    max_actual = np.max(actual)
    min_actual = np.min(actual)

    range_actual = max_actual - min_actual

    return rmsd / range_actual


def calculate_scores(model_class):
    # Training Scores
    model_class.train_rmse = rmse(model_class.y_train, model_class.y_hat_train)
    model_class.train_mse = mse(model_class.y_train, model_class.y_hat_train)
    model_class.train_r_squared = r_squared(model_class.y_train, model_class.y_hat_train)
    model_class.train_mae = mae(model_class.y_train, model_class.y_hat_train)
    model_class.train_nnse = nnse(model_class.y_train, model_class.y_hat_train)
    model_class.train_r = r(model_class.y_train, model_class.y_hat_train)
    model_class.train_ia = ia(model_class.y_train, model_class.y_hat_train)
    model_class.train_kge = kge(model_class.y_train, model_class.y_hat_train)
    model_class.train_mbe = mbe(model_class.y_train, model_class.y_hat_train)
    model_class.train_nrmse = nrmse(model_class.y_train, model_class.y_hat_train)

    # Testing Scores
    model_class.test_rmse = rmse(model_class.y_test, model_class.y_hat_test)
    model_class.test_mse = mse(model_class.y_test, model_class.y_hat_test)
    model_class.test_r_squared = r_squared(model_class.y_test, model_class.y_hat_test)
    model_class.test_mae = mae(model_class.y_test, model_class.y_hat_test)
    model_class.test_nnse = nnse(model_class.y_test, model_class.y_hat_test)
    model_class.test_r = r(model_class.y_test, model_class.y_hat_test)
    model_class.test_ia = ia(model_class.y_test, model_class.y_hat_test)
    model_class.test_kge = kge(model_class.y_test, model_class.y_hat_test)
    model_class.test_mbe = mbe(model_class.y_test, model_class.y_hat_test)
    model_class.test_nrmse = nrmse(model_class.y_test, model_class.y_hat_test)

    if model_class.gwl_output not in ['delta', 'average_delta']:
        model_class.train_mape = mape(model_class.unscaled_y_train, model_class.unscaled_y_hat_train)  # Uses unscaled to avoid division by 0
        model_class.test_mape = mape(model_class.unscaled_y_test, model_class.unscaled_y_hat_test)  # Uses unscaled to avoid division by 0

        model_class.train_scores_dict = {'Train Root Mean Squared Error': model_class.train_rmse,
                                         'Train Mean Squared Error': model_class.train_mse,
                                         'Train Normalised Root Mean Squared Error': model_class.train_nrmse,
                                         'Train Coefficient of Determination': model_class.train_r_squared,
                                         'Train Normalised Nash Sutcliffe Efficiency': model_class.train_nnse,
                                         'Train Mean Absolute Error': model_class.train_mae,
                                         "Train Pearson's Correlation Coefficient": model_class.train_r,
                                         'Train Index of Agreement': model_class.train_ia,
                                         'Train Kling-Gupta Efficiency': model_class.train_kge,
                                         'Train Mean Bias Error': model_class.train_mbe,
                                         'Train Mean Absolute Percentage Error': model_class.train_mape}

        model_class.test_scores_dict = {'Test Root Mean Squared Error': model_class.test_rmse,
                                        'Test Mean Squared Error': model_class.test_mse,
                                        'Test Normalised Root Mean Squared Error': model_class.test_nrmse,
                                        'Test Coefficient of Determination': model_class.test_r_squared,
                                        'Test Normalised Nash Sutcliffe Efficiency': model_class.test_nnse,
                                        'Test Mean Absolute Error': model_class.test_mae,
                                        "Test Pearson's Correlation Coefficient": model_class.test_r,
                                        'Test Index of Agreement': model_class.test_ia,
                                        'Test Kling-Gupta Efficiency': model_class.test_kge,
                                        'Test Mean Bias Error': model_class.test_mbe,
                                        'Test Mean Absolute Percentage Error': model_class.test_mape}
    else:
        # No MAPE since
        model_class.train_scores_dict = {'Train Root Mean Squared Error': model_class.train_rmse,
                                         'Train Mean Squared Error': model_class.train_mse,
                                         'Train Normalised Root Mean Squared Error': model_class.train_nrmse,
                                         'Train Coefficient of Determination': model_class.train_r_squared,
                                         'Train Normalised Nash Sutcliffe Efficiency': model_class.train_nnse,
                                         'Train Mean Absolute Error': model_class.train_mae,
                                         "Train Pearson's Correlation Coefficient": model_class.train_r,
                                         'Train Index of Agreement': model_class.train_ia,
                                         'Train Kling-Gupta Efficiency': model_class.train_kge,
                                         'Train Mean Bias Error': model_class.train_mbe}

        model_class.test_scores_dict = {'Test Root Mean Squared Error': model_class.test_rmse,
                                        'Test Mean Squared Error': model_class.test_mse,
                                        'Test Normalised Root Mean Squared Error': model_class.test_nrmse,
                                        'Test Coefficient of Determination': model_class.test_r_squared,
                                        'Test Normalised Nash Sutcliffe Efficiency': model_class.test_nnse,
                                        'Test Mean Absolute Error': model_class.test_mae,
                                        "Test Pearson's Correlation Coefficient": model_class.test_r,
                                        'Test Index of Agreement': model_class.test_ia,
                                        'Test Kling-Gupta Efficiency': model_class.test_kge,
                                        'Test Mean Bias Error': model_class.test_mbe}
