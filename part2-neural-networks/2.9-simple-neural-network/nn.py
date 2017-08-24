import numpy as np

#implementing sigmoid function
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

#calculate the output
output = sigmoid(np.dot(inputs,weights) + bias)

print('Output:')
print(output)
