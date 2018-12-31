# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:21:37 2018

"""
import numpy as np
from sklearn.utils import shuffle
import sys
import pickle

DATA_DIR = 'data'
TRAIN = '/letters.train.data'
TEST = '/letters.test.data'
START = '$'

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

    @staticmethod
    def update(weights, example, y_hat, y_actual):
        if y_hat != y_actual:
            weights[y_actual] += example
            weights[y_hat] -= example
        return weights

    @staticmethod
    def predict(weights, example):
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
        """The code in comments is equivalent to the np code used
        maximum = float('-inf')
        argmax = None
        for i in range(self.num_classes):
            calc = weights[i].dot(self.phi(example, i))
            if calc > maximum:
                maximum = calc
                argmax = i
        return argmax """
        phi_matrix = [self.phi(example, i) for i in range(self.num_classes)]
        return np.argmax(np.einsum('ij,ij->i', phi_matrix, weights))
    
    def update(self, weights, example, y_hat, y_actual):
        if y_hat != y_actual:
            weights[y_actual] += self.phi(example, y_actual)
            weights[y_hat] -= self.phi(example, y_hat)
        return weights
        

class Bigram_structured_perceptron:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def initialize_weights(self):
        return np.zeros([self.num_classes, IMG_SIZE * self.num_classes + (self.num_classes ** 2)])

    def phi(self, x, y, prev_y):
        word_phi = np.zeros(IMG_SIZE * self.num_classes)
        word_phi[y: y + IMG_SIZE] = x
        bigram_phi = np.zeros([self.num_classes, self.num_classes])
        bigram_phi[prev_y][y] = 1
        bigram_phi.flatten()
        return np.concatenate((word_phi, bigram_phi))

    def predict(self, W, x):
        label_size = len(classes) # ???
        D_S = np.zeros([label_size, len(classes)]) # scores
        D_PI = np.zeros([label_size, len(classes)]) # back-pointers

        # Initialization
        for i, tag in enumerate(classes.keys()):
            phi = self.phi(x, tag, prev_y=START)
            s = W * phi
            D_S[0][i] = s
            D_PI[0][i] = 0

        # Recursion
        for i in range(1, label_size):
            for j, tag in enumerate(classes.keys()):
                curr_char = tag # ???
                # d_best = i_best = -1
                d_best, i_best = max(W * phi(x,y * curr_char) + D_S[i - 1][y] for y in classes.values())
                D_S[i][j] = d_best
                D_PI[i][j] = i_best

        # Back-track
        y_hat = np.zeros(label_size)
        d_best = -1
        for i, tag in enumerate(classes.keys()):
            if d_best < D_S[label_size - 1][i]:
                y_hat[label_size - 1] = i
                d_best = D_S[label_size - 1][i]

        for i in range(label_size)[label_size - 2: -1:]:
            y_hat[i] = D_PI[i + 1][y_hat[i + 1]]
        return y_hat

    def update(self, weights, example, y_hat, y_actual):
        if y_hat != y_actual:
            weights[y_actual] += self.phi(example, y_actual)
            weights[y_hat] -= self.phi(example, y_hat)
        return weights


def perceptron(x, y, method):
    weights = method.initialize_weights()
    total = len(x)
    final_weights = []
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch #{epoch}')
        weights_adj = []
        x, y = shuffle(x, y, random_state=epoch)
        for index, (example, tag) in enumerate(zip(x, y)):
            if index % 10000 == 0:
                print(f'-- {round(float(index) / total * 100, 2)}%')
            prediction = method.predict(weights, example)
            if prediction != classes[tag]:
                weights = method.update(weights, example, prediction, classes[tag])
                weights_adj.append(weights)
        final_weights.append(np.average(weights_adj, 0))
    weights = np.average(final_weights, 0)
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
    #method_type = sys.argv[1]
    method_type = 'b'
    data_x, data_y = load_data(TRAIN)
    cutoff = round(len(data_x) * 0.8)
    classes = {v: i for i, v in enumerate(set(data_y))}
    ONE_HOT = np.eye(len(classes))
    train_x, train_y = (data_x[:cutoff], data_y[:cutoff])
    pred_x, pred_y = (data_x[cutoff:], data_y[cutoff:])

    if method_type == 'a':
        method = Multiclass_perceptron(len(classes))

    elif method_type == 'b':
        method = Structured_perceptron(len(classes))

    elif method_type == 'c':
        classes[START] = len(classes)
        method = Bigram_structured_perceptron(len(classes))

    else:
        raise Exception("Illegal command line arguments")

    model = perceptron(train_x, train_y, method)
    pickle.dump(model, open('model_' + method_type, 'rb'))
    test(pred_x, pred_y, model, method)
