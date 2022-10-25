# Program multi neuron batch input
# 1.  Input layer feature 10
# 2. Per batch nya 6 input
# 3. Hidden layer 1, 5 neuron
# 4. Hidden layer 2, 3 neuron

# inisialisasi numpy
import numpy as np

# inisialisasi variabel inputs
inputs = [
    [2.5, 2.5, 3.0, 1.5, 3.5, 2.5, 4.0, 3.5, 5.5, 5.6],
    [2.5, 3.4, 1.2, 2.4, 4.2, 2.4, 5.2, 3.4, 4.2, 6.4],
    [3.7, 14.5, 14.0, 15.5, 30.5, 20.5, 40.5, 30.5, 50.5, 60.5],
    [4.7, 2.8, 3.6, 2.5, 3.5, 4.8, 4.8, 5.8, 5.0, 6.8],
    [5.5, 7.4, 7.4, 7.5, 9.2, 7.4, 5.2, 7.4, 10.5, 8.4],
    [11.5, 15.4, 17.3, 15.4, 19.2, 18.4, 20.2, 20.4, 10.5, 10.4],
]
# Hidden layer 1
# inisialisasi bobot hidden layer 1
weights = [
    [2.0, 2.5, 1.0, 2.6, 3.5, 3.6, 4.5, 4.2, 6.0, 4.5],
    [2.5, 2.4, 3.2, 3.4, 3.5, 2.4, 3.2, 4.5, 6.2, 6.4],
    [3.7, 2.8, 2.5, 3.8, 3.5, 4.8, 4.5, 5.8, 5.7, 5.9],
    [3.5, 6.5, 8.2, 7.5, 9.2, 8.5, 9.5, 9.5, 10.5, 10.2],
    [3.6, 19.5, 19.0, 20.7, 30.6, 20.5, 40.5, 20.5, 40.0, 40.5]
]
# inisialisasi bias hidden layer 1
biases = [2.5, 2.2, 4.1, 5.7, 5.0]

#output
outputs = np.dot(inputs, np.array(weights) . T) + biases

# hidden layer 2
# inisialisasi bobot hidden layer 2
weights2 = [
    [-2.0, 2.1, 2.5, 1.0, 1.5],
    [-1.0, 2, 0.5, 1.5, 2.0],
    [1.5, 3, 2.0, 1.0, 2.1]
]
# inisialisasi bias hidden layer 2
biases2 = [2.0, 1.5, 3.0]

#output2
outputs2 = np.dot(outputs, np.array(weights2).T) + biases2

#cetak output
print(outputs)
print(outputs2)