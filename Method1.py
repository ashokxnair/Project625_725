# --------------------------------------------------------------------
# Program to test the Linear Regression class using the classical
#   Stochastic Gradient Descent algorithm
#
# Author: Ashok Nair
#
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import LinearRegression
import Utilities

# Index value of the target or class type
TARGET_INDEX = 13

boston_df = pd.read_csv('Boston.csv', header=0)

print('Data Size : ', boston_df.shape)

# Prepare the data (Data wrangling)
# Shuffle the dataframe to ensure the target class is mixed
boston_df = boston_df.sample(frac=1, random_state=12).reset_index(drop=True)

# Create output file with current date and time in name
# get current date and time
timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = "output/" + timestr
print('File name : ', file_name + '_Method1.txt')
output_file = open(file_name + '_Method1.txt', 'w')
output_file.write('Multivariable Linear Regression (Method 1)\n')
output_file.write('Results of Boston House Price Data\n\n')
output_file.write('Classical Stochastic process with constant step size and '
                  'use all observations\n\n')

# Convert the dataframe to array
boston_array = boston_df.to_numpy()

# Separate the independent and dependent variables
y_array = boston_array[:, TARGET_INDEX]
x_array = boston_array[:, :-1]

# Instantiate the Linear regression object
ln = LinearRegression.LinearRegression(len(x_array[0]))

# Standardize the data and set the values
X_transform = Utilities.standardize(x_array)
ln.set_values(X_transform, y_array)

step_size = 0.00001
epochs = 10000

start_time = time.time()
coef_estimates = ln.coef_estimator_sgd(step_size, epochs)
end_time = time.time()
# get the execution time
elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

output_file.write('\nStep Size :' + str(step_size))
output_file.write('\nEpochs : ' + str(epochs))
output_file.write('\n\nCoefficient estimates : \n' + str(coef_estimates[1:]))
output_file.write('\n\nExecution time : %0.2f seconds\n' % elapsed_time)

cost_lst = ln.get_cost_lst()
plt.plot(np.arange(1, len(cost_lst)), cost_lst[1:])
plt.title('Cost function Graph')
plt.xlabel('Number of iterations (Epoch)')
plt.ylabel('Cost')
# plt.show()

plt.savefig(file_name + '_Method1.png')

output_file.close()

print('Processing complete')
