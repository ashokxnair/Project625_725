# --------------------------------------------------------------------
# This file contains the standardize function
#
# Author: Ashok Nair
#
# --------------------------------------------------------------------

import numpy as np


def standardize(data_array):
    """
    Function to standardize the data
    :param data_array: The data to standardize
    :type data_array: Array
    :return: Standardized data
    :rtype: Array
    """
    result_array = np.empty(shape=(data_array.shape[0], data_array.shape[1]))
    for row in range(data_array.shape[0]):
        for column in range(data_array.shape[1]):
            if column == data_array.shape[1] - 1:
                result_array[row][column] = data_array[row][column]
                continue
            result_array[row][column] = (data_array[row][column] - np.mean(data_array[:, column])) \
                                        / np.std(data_array[:, column])
    return result_array
