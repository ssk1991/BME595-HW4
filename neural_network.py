import numpy as np
import math
from random import randint


class NeuralNetwork:

    def __init__(self, layer_sizes=[]):
        self.layer_sizes = layer_sizes
        self.dE_dTheta = {}
        self.Theta = {}
        self.Z = {}
        # A has the biases
        self.A = {}
        self.L = len(layer_sizes)

        np.random.seed(1)
        for i in range(0, len(self.layer_sizes) - 1):
            random_normal_vector = np.random.normal(0, 1 / math.sqrt(self.layer_sizes[i] * self.layer_sizes[i + 1]), (1 + self.layer_sizes[i]) * self.layer_sizes[i + 1])
            random_normal_matrix = np.reshape(random_normal_vector, (1 + self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.Theta[i] = random_normal_matrix

    def getLayer(self, layer):
        return self.Theta[layer]

    def forward(self, input):

        a = input

        for i in range(0, self.L - 1):
            bias = np.ones(a.shape[1])
            a = np.append(a, bias)
            a = np.reshape(a, (int(len(a) / input.shape[1]), input.shape[1]))
            self.A[i] = a
            z = np.dot(self.Theta[i].T, a)
            self.Z[i + 1] = z
            a = 1 / (1 + np.exp(-z))
            # print(i)
        self.A[self.L - 1] = a
        return np.round_(a, 0)
        # print("A", self.A)
        # print("Z", self.Z)

    def backward(self, target):

        const_product = (self.A[self.L - 1] - target) * self.A[self.L - 1] * (1 - self.A[self.L - 1])
        self.dE_dTheta[self.L - 2] = const_product * self.A[self.L - 2]

        # Code only works for 0 or 1 hidden layers
        for layer in reversed(range(0, self.L - 2)):

            theta_times_A = self.Theta[layer + 1] * self.A[layer + 1] * (1 - self.A[layer + 1])
            # Backprop algorithm requires last row of the above product to be removed
            theta_times_A = theta_times_A[0: len(theta_times_A) - 1, :]

            self.dE_dTheta[layer] = const_product * np.dot(self.A[layer], theta_times_A.T)

            const_product = const_product * np.dot(self.Theta[layer + 1], theta_times_A.T)

    def updateParams(self, eta):

        for layer in range(0, self.L - 1):
            self.Theta[layer] -= self.dE_dTheta[layer]


if __name__ == "__main__":
    model = NeuralNetwork([2, 2, 1])
    # x = np.array([[1],[1]])
    # target = 0.2
    for epoch in range(0, 10000):
        x1 = randint(0, 1)
        x2 = randint(0, 1)
        y = model.forward(np.array([[x1], [x2]]))
        target = x1 != x2
        print("epoch number", epoch, "predicted value", y, "loss", y - target)
        model.backward(target)
        model.updateParams(0.4)
    # print(model.getLayer(0))
