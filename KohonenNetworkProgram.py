'''
Created on September 29, 2021

@author: Syed.Ausaf.Hussain
'''

import numpy as np
import pandas


# calculate Distance
def calculate(row, weights):
    activation = []
    for i in range(len(weights)):
        activation.append(i)
        for j in range(len(row)):
            activation[i] = activation[i] + ((row[j] - weights[i][j])**2)
        activation[i] = abs(activation[i])
    return activation.index(min(activation))

# Estimate Perceptron weights using Kohonen
def train_weights(train, weights, l_rate, n_epoch):
    for epoch in range(n_epoch):
        targetAchived = False
        for row in train:
            winnerIndex = calculate(row, weights)
            for w in range(len(weights)):
                d = row[w] - weights[winnerIndex][w]
                if(abs(d) < 0.01):
                    targetAchived = True
                else:
                    targetAchived = False
                weights[winnerIndex][w] = weights[winnerIndex][w] + l_rate*(d)
        if targetAchived:
            break
    return weights


data = pandas.read_csv('data-Kohonen.csv')
print("Data\n", data)

#weight/prototype
weights = np.array([[7.0, 2.0], [2.0, 9.0]])

# Learning Rate (step size)
c = 0.5

print('Initial Weights: ', weights)
print('Learning Rate  (step size): ' + str(c))

print("Final weights", train_weights(data.values.tolist(), weights, c, n_epoch=200))
