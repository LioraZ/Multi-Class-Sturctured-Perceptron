# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:21:37 2018

"""
import numpy as np
from sklearn.utils import shuffle
import sys
import pickle
from collections import defaultdict

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
        letter_pixels = []
        letter_tags = []
        for line in file:
            fields = line.split()
            pixels = np.array([int(x) for x in fields[6:]])
            letter_pixels.append(pixels)
            letter_tags.append(fields[LETTER])
            if fields[NEXT_ID] == '-1':
                data.append([letter_pixels, letter_tags])
                tags.extend(letter_tags)
                letter_pixels = []
                letter_tags = []
    return data, tags


class Bigram_structured_perceptron:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def initialize_weights(self):
        return np.zeros(IMG_SIZE * self.num_classes + (self.num_classes ** 2))

    def phi(self, x, y, prev_y):
        # y, prev_y = classes[y], classes[prev_y]
        word_phi = np.zeros(IMG_SIZE * self.num_classes)
        word_phi[y: y + IMG_SIZE] = x
        bigram_phi = np.zeros([self.num_classes, self.num_classes])
        bigram_phi[prev_y][y] = 1
        bigram_phi = np.ravel(bigram_phi)
        return np.concatenate((word_phi, bigram_phi))

    def predict(self, W, words):
        label_size = len(words)
        # D_S = np.zeros([label_size, len(classes)]) # scores
        # D_PI = np.zeros([label_size, len(classes)]) # back-pointers
        D_S = defaultdict(int)
        D_PI = defaultdict()

        # Initialization
        for i, tag in enumerate(classes.keys()):
            phi = self.phi(words[0], classes[tag], prev_y=classes[START])
            s = np.dot(W, phi)
            D_S[(0, i)] = s
            D_PI[(0, i)] = 0

        # Recursion
        for i in range(1, label_size):
            for j, tag in enumerate(classes.keys()):
                temp_vals = [np.dot(W, self.phi(words[i], classes[tag], classes[prev_y])) + D_S[(i - 1, prev_y)] for
                             prev_y in classes.keys()]
                D_PI[(i, j)] = np.argmax(temp_vals)
                D_S[(i, j)] = temp_vals[D_PI[(i, j)]]

        # Back-track
        y_hat = np.zeros(label_size, dtype=int)
        d_best = -1
        for i, tag in enumerate(classes.keys()):
            if d_best < D_S[(label_size - 1, i)]:
                y_hat[label_size - 1] = i
                d_best = D_S[(label_size - 1, i)]

        for i in range(label_size)[label_size - 2: -1:]:
            y_hat[i] = D_PI[(i + 1, y_hat[i + 1])]
        return y_hat

    """def update(self, weights, example, y_hat, y_actual, y_prev):
        if y_hat != y_actual:
            weights += self.phi(example, y_actual, y_prev)
            weights -= self.phi(example, y_hat, y_prev)
        return weights"""
    def update(self, weights, letters, tag_vec, predictions):
        prev_tag= prev_pred = classes[START]
        real_phi = np.zeros(IMG_SIZE * self.num_classes + (self.num_classes ** 2))
        pred_phi = np.zeros(IMG_SIZE * self.num_classes + (self.num_classes ** 2))
        for i, (tag, prediction) in enumerate(zip(tag_vec, predictions)):
            real_phi += self.phi(letters[i], tag, prev_tag)
            pred_phi += self.phi(letters[i], prediction, prev_pred)
            prev_tag = tag
            prev_pred = prediction
        weights += real_phi - pred_phi
        return weights


def perceptron(x, method):
    weights = method.initialize_weights()
    total = len(x)
    final_weights = []
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch #{epoch}')
        weights_adj = []
        x = shuffle(x, random_state=epoch)
        for index, word in enumerate(x):
            letters, letter_tags = word
            if index % 500 == 0:
                print(f'-- {round(float(index) / total * 100, 2)}%')
            # prev_tag = classes[START]
            predictions = method.predict(weights, letters)
            tag_vec = [classes[tag] for tag in letter_tags]
            weights = method.update(weights, letters, tag_vec, predictions)
            weights_adj.append(weights)
            """for i, (tag, prediction) in enumerate(zip(tag_vec, predictions)):
                if prediction != tag:
                    weights = method.update(weights, letters[i], prediction, tag, prev_tag)
                    weights_adj.append(weights)
                prev_tag = tag"""
            """for i, (letter, tag) in enumerate(zip(letters, letter_tags)):
                prediction = method.predict(weights, letter, len(letters))
                if prediction[i] != classes[tag]:
                    weights = method.update(weights, letters, prediction[i], classes[tag], classes[prev_tag])
                    weights_adj.append(weights)
                prev_tag = tag"""
        final_weights.append(np.average(weights_adj, 0))
        test(x, np.average(final_weights, 0), method)
    weights = np.average(final_weights, 0)
    return weights


def test(x, weights, method):
    correct = 0.0
    incorrect = 0.0
    for word in x:
        letters, letter_tags = word
        predictions = method.predict(weights, letters)
        tag_vec = [classes[tag] for tag in letter_tags]
        for tag, prediction in zip(tag_vec, predictions):
            if prediction == tag:
                correct += 1.0
            else:
                incorrect += 1.0

    total = correct + incorrect
    success_rate = round(correct / total * 100, 3)
    print(f'Got {success_rate}% correct.')
    return success_rate


if __name__ == '__main__':
    #method_type = sys.argv[1]
    method_type = 'c'
    data, tags = load_data(TRAIN)
    cutoff = round(len(data) * 0.8)
    classes = {v: i for i, v in enumerate(set(tags).union(START))}
    ONE_HOT = np.eye(len(classes))
    train_x = data[:cutoff]
    pred_x = data[cutoff:]

    if method_type == 'c':
        # classes[START] = len(classes)
        method = Bigram_structured_perceptron(len(classes))

    else:
        raise Exception("Illegal command line arguments")

    model = perceptron(train_x, method)
    pickle.dump(model, open('model_' + method_type, 'wb'))
    test(pred_x, model, method)
