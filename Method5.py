# --------------------------------------------------------------------
# Program to test the Linear Regression class using the Averaged
#   Stochastic Gradient Descent algorithm with variable step size
#   and all observations
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
print('File name : ', file_name + '_Method5.txt')
output_file = open(file_name + '_Method5.txt', 'w')
output_file.write('Multivariable Linear Regression (Method 5)\n')
output_file.write('Results of Boston House Price Data - Method 3\n\n')
output_file.write('Averaged Stochastic process with variable step size and '
                  'use all observations\n\n')

# Convert the dataframe to array
boston_array = boston_df.to_numpy()

y_array = boston_array[:, TARGET_INDEX]
x_array = boston_array[:, :-1]

ln = LinearRegression.LinearRegression(len(x_array[0]))

X_transform = Utilities.standardize(x_array)
ln.set_values(X_transform, y_array)

step_size = [0.001, 0.0001, 0.00001, 0.000001]
step_decider = [500, 2000, 10000]

output_file.write("\n\nStep Size: " + str(step_size))
output_file.write("\nStep Decider: " + str(step_decider))

start_time = time.time()
coef_estimates = ln.coef_estimator_sgd_varstep(step_size, step_decider, 10000, averaged=True)
end_time = time.time()
# get the execution time
elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

output_file.write("\n\n Coefficient estimates :\n" + str(coef_estimates[1:]))
output_file.write('\n\n Execution time : %0.2f seconds\n' % elapsed_time)

cost_lst = ln.get_cost_lst()
plt.plot(np.arange(1, len(cost_lst)), cost_lst[1:])
plt.title('Cost function Graph')
plt.xlabel('Number of iterations (Epoch)')
plt.ylabel('Cost')
plt.savefig(file_name + '_Method5.png')

output_file.close()

print('Processing complete')
