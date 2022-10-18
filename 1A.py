import numpy as np

inputs = [2.0, 3.5, 1.0, 1.5, 2.0, 2.5, 3.0, 1.5, 2.0, 4.5]
weights = [3.2, 2.4, 1.6, 4.8, 2.0, 1.2, 2.4, 2.6, 3.8, 4.0]

bias = 5.0

outputs = np.dot(weights, inputs) + bias
print(outputs)