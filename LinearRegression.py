# --------------------------------------------------------------------
# This file contains the Linear regression algorithm
#
# Author: Ashok Nair
#
# --------------------------------------------------------------------

import numpy as np


class LinearRegression:

    def __init__(self, col_length):
        """
        Method to initialize the object
        :param col_length: The number of independent variables in the data
        :type data_array: Integer
        :return: None
        """
        self.y_values = None
        self.x_values = None
        np.random.seed(2)

        # Initialize theta with random values
        self.theta = np.random.randn(col_length + 1, 1)
        self. cost_lst = []
        self.coef_estimates = []

    def set_values(self, x_values, y_values):
        """
        Method to set the X values and the Y values
        :param x_values: Array containing the X data
        :param y_values: Array containing the target values
        :return: None
        """
        self.x_values = x_values
        self.y_values = y_values

    def get_cost_lst(self):
        """
        Method that returns the cost values for plotting
        :return: Array of cost values
        """
        return self.cost_lst

    def add_bias(self):
        """
        Method to add bias to the X values
        :return: None
        """
        self.x_values = np.c_[np.ones((len(self.x_values), 1)), self.x_values]

    def calc_cost(self, m, y_hat):
        """
        Calculate the cost
        :param m: The number of observations
        :param y_hat: The estimated values of Y
        :return: Cost value
        """
        return 1 / (2 * m) * ((y_hat - self.y_values) ** 2)

    def coef_estimator_sgd(self, step_size, epochs, averaged=False, tolerance=1e-06):
        """
        Method to estimate the coefficients
        :param step_size: Step size
        :param epochs: The number of iterations
        :param averaged: Set to True to use averaged SGD
        :param tolerance: The accuracy level
        :return: The estimated coefficients
        """
        y_new = np.reshape(self.y_values, (len(self.y_values), 1))

        m = len(self.x_values)

        # If the coef_estimates exists use it
        if len(self.coef_estimates):
            theta_hat = self.coef_estimates
        else:
            theta_hat = self.theta

        self.add_bias()

        # Iterate for the number of epochs
        for _ in range(epochs):
            gradients = 2 / m * self.x_values.T.dot(self.x_values.dot(theta_hat) - y_new)

            prev_theta_hat = theta_hat
            if averaged:
                theta_hat = np.average(theta_hat) # The average for theta_hat
            theta_hat = theta_hat - step_size * gradients
            y_hat = self.x_values.dot(theta_hat)
            cost_value = self.calc_cost(m, y_hat)  # Calculate the loss for each training instance
            total = 0
            for i in range(len(self.y_values)):
                total += cost_value[i][0]  # Calculate the cost function for each iteration
            self.cost_lst.append(total)
            difference = prev_theta_hat - theta_hat
            if np.all(np.abs(difference) <= tolerance):
                break
        self.coef_estimates = theta_hat
        return theta_hat

    def coef_estimator_sgd_varstep(self, step_size, step_decider, epochs, averaged=False, tolerance=1e-06):
        """
        Method to estimate coefficients using variable step
        :param step_size: Step size array
        :param step_decider: iteration number to change the step size
        :param epochs: Number of iterations
        :param averaged: Uses averaged SGD when set to True
        :param tolerance: The accuracy level
        :return: Estimated coefficients
        """

        y_new = np.reshape(self.y_values, (len(self.y_values), 1))
        self.cost_lst = []
        m = len(self.x_values)
        if len(self.coef_estimates):
            theta_hat = self.coef_estimates
        else:
            theta_hat = self.theta
        self.add_bias()
        step_size_index = 0
        for index1 in range(epochs):
            gradients = 2 / m * self.x_values.T.dot(self.x_values.dot(theta_hat) - y_new)

            prev_theta_hat = theta_hat
            if averaged:
                theta_hat = np.average(theta_hat)
            theta_hat = theta_hat - step_size[step_size_index] * gradients
            if index1 >= step_decider[step_size_index]:
                step_size_index += 1

            y_hat = self.x_values.dot(theta_hat)
            cost_value = self.calc_cost(m, y_hat)  # Calculate the loss for each training instance
            total = 0
            for i in range(len(self.y_values)):
                total += cost_value[i][0]  # Calculate the cost function for each iteration
            self.cost_lst.append(total)
            difference = prev_theta_hat - theta_hat
            if np.all(np.abs(difference) <= tolerance):
                break

        return theta_hat
