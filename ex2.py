# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:21:37 2018

"""

import numpy as np
from sklearn.utils import shuffle

DATA_DIR = 'data'
TRAIN = '/letters.train.data'
TEST = '/letters.test.data'
START = '<s>'

ID = 0
LETTER = 1
NEXT_ID = 2
FOLD = 3
PIXELS = 4

IMG_SIZE = 128
EPOCHS = 3
ONE_HOT = ''
CLASSES = ''


def load_data(data_path):
    data = []
    tags = []
    with open(DATA_DIR + data_path, 'r') as file:
        for line in file:
            fields = line.split()
            pixels = np.array([int(x) for x in fields[6:]])
            data.append(pixels)
            tags.append(fields[LETTER])
    return data, tags

class Multiclass_perceptron:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def initialize_weights(self):
        return np.random.rand(self.num_classes, IMG_SIZE)
    
    def update(self, weights, example, y_hat, y_actual):
        if y_hat != y_actual:
            weights[y_actual] += example
            weights[y_hat] -= example
        return weights
    
    def predict(self, weights, example):
        return np.argmax(weights.dot(example))

class Structured_perceptron:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def initialize_weights(self):
        return np.zeros([self.num_classes, IMG_SIZE * self.num_classes])
    
    def phi(self, x, y):
        end = np.concatenate((x, np.zeros((self.num_classes - y - 1) * IMG_SIZE)))
        return np.concatenate((np.zeros(y * IMG_SIZE), end))
    
    def predict(self, weights, example):
#        maximum = float('-inf')
#        argmax = None
#        for i in range(self.num_classes):
#            calc = weights[i].dot(self.phi(example, i))
#            if calc > maximum:
#                maximum = calc
#                argmax = i
#        return argmax
        phi_matrix = [self.phi(example, i) for i in range(self.num_classes)]
        return np.argmax(np.einsum('ij,ij->i', phi_matrix, weights))
    
    def update(self, weights, example, y_hat, y_actual):
        if y_hat != y_actual:
            weights[y_actual] += self.phi(example, y_actual)
            weights[y_hat] -= self.phi(example, y_hat)
        return weights
        


def perceptron(x, y, method):
    weights = method.initialize_weights()
    total = len(x)
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch #{epoch}')
        x, y = shuffle(x, y, random_state=epoch)
        weights_adj = []
        for index, (example, tag) in enumerate(zip(x, y)):
            if index % 10000 == 0:
                print(f'-- {round(float(index) / total * 100, 2)}%')
            prediction = method.predict(weights, example)
            if prediction != classes[tag]:
                weights = method.update(weights, example, prediction, classes[tag])
                weights_adj.append(weights)
        weights = np.average(weights_adj, 0)
    return weights

def test(x, y, weights, method):
    correct = 0.0
    incorrect = 0.0
    for (example, tag) in zip(x, y):
        prediction = method.predict(weights, example)
        if prediction == classes[tag]:
            correct += 1.0
        else:
            incorrect += 1.0
    total = correct + incorrect
    success_rate = round(correct / total * 100, 3)
    print(f'Got {success_rate}% correct.')
    return success_rate
    
        


if __name__ == '__main__':
    data_x, data_y = load_data(TRAIN)
    cutoff = round(len(data_x) * 0.8)
    classes = {v: i for i, v in enumerate(set(data_y))}
    ONE_HOT = np.eye(len(classes))
    train_x, train_y = (data_x[:cutoff], data_y[:cutoff])
    pred_x, pred_y = (data_x[cutoff:], data_y[cutoff:])
    
    method = Multiclass_perceptron(len(classes))
    model = perceptron(train_x, train_y, method)
    test(pred_x, pred_y, model, method)
