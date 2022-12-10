# --------------------------------------------------------------------
# Program to test the Linear Regression class using the Classical
#   Stochastic Gradient Descent algorithm with variable step size
#   and varying number of observations per step
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
print('File name : ', file_name + '_Method2.txt')
output_file = open(file_name + "_Method2.txt", 'w')
output_file.write('Multivariable Linear Regression (Method 2)\n')
output_file.write('Results of Boston House Price Data\n\n')
output_file.write('Classical Stochastic process with variable step size and '
                  'varying number of observations per step\n\n')


# Convert the dataframe to array
boston_array = boston_df.to_numpy()

# Separate the target variable
y_array = boston_array[:, TARGET_INDEX]
x_array = boston_array[:, :-1]

# Spilt the data into 3 chunks
X_values = np.array_split(x_array, [int(0.4 * len(x_array)), int(0.6 * len(x_array))])
y_values = np.array_split(y_array, [int(0.4 * len(y_array)), int(0.6 * len(y_array))])

ln = LinearRegression.LinearRegression(len(X_values[0][0]))
time_sum = 0
for index1 in range(3):
    print("Processing batch: ", index1)
    X_transform = Utilities.standardize(X_values[index1])
    ln.set_values(X_transform, y_values[index1])
    step_size = [0.001, 0.0001, 0.00001, 0.000001]
    step_decider = [500, 2000, 10000]

    output_file.write("\n\nStep Size: " + str(step_size))
    output_file.write("\nStep Decider: " + str(step_decider))

    start_time = time.time()
    coef_estimates = ln.coef_estimator_sgd_varstep(step_size, step_decider, 10000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_sum += elapsed_time

    output_file.write("\n\n Coefficient estimates : \n" + str(coef_estimates[1:]))

print('Execution time:', time_sum, 'seconds')
output_file.write('\n\n Execution time : %0.2f seconds\n' % time_sum)
cost_lst = ln.get_cost_lst()

plt.plot(np.arange(1, len(cost_lst)), cost_lst[1:])
plt.title('Cost function Graph')
plt.xlabel('Number of iterations (Epoch)')
plt.ylabel('Cost')
# plt.savefig(file_name + '_' + str(index1) + '.png')
plt.savefig(file_name + '_Method2.png')

output_file.close()

print('Processing complete')
