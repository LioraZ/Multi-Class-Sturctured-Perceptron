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
NUM_CLASSES = 27


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
        word_phi = np.zeros(IMG_SIZE * self.num_classes)
        bigram_phi = np.zeros([self.num_classes, self.num_classes])
        word_phi[y * IMG_SIZE: (y + 1) * IMG_SIZE] = x
        bigram_phi[prev_y][y] = 1
        bigram_phi = np.ravel(bigram_phi)
        return np.concatenate((word_phi, bigram_phi))

    def predict(self, W, words):
        label_size = len(words)
        D_S = defaultdict(float) # scores
        D_PI = defaultdict(int) # back-pointers

        # Initialization
        for i, tag in enumerate(classes.keys()):
            phi = self.phi(words[0], classes[tag], prev_y=classes[START])
            s = np.dot(W, phi)
            D_S[(0, classes[tag])] = s
            D_PI[(0, classes[tag])] = 0

        # Recursion
        for i in range(1, label_size):
            for j, tag in enumerate(classes.keys()):
                temp_vals = [np.dot(W, self.phi(words[i], classes[tag], classes[prev_y])) + D_S[(i - 1, classes[prev_y])] for
                             prev_y in classes.keys()]
                D_PI[(i, classes[tag])] = np.argmax(temp_vals)
                D_S[(i, classes[tag])] = temp_vals[D_PI[(i, classes[tag])]]

        # Back-track
        y_hat = np.zeros(label_size, dtype=int)
        d_best = -1
        for i, tag in enumerate(classes.keys()):
            if d_best < D_S[(label_size - 1, classes[tag])]:
                y_hat[label_size - 1] = classes[tag]
                d_best = D_S[(label_size - 1, classes[tag])]

        for i in range(label_size - 2, -1, -1):
            y_hat[i] = D_PI[(i + 1, y_hat[i + 1])]
        return y_hat

    def update(self, weights, letters, tag_vec, predictions):
        prev_tag = prev_pred = classes[START]
        bigram_phi = np.zeros((self.num_classes, self.num_classes))
        pixels_phi = np.zeros(self.num_classes * IMG_SIZE)
        for i, (tag, prediction) in enumerate(zip(tag_vec, predictions)):
            pixels_phi[tag * IMG_SIZE: (tag + 1) * IMG_SIZE] += letters[i]
            pixels_phi[prediction * IMG_SIZE: (prediction + 1) * IMG_SIZE] -= letters[i]
            bigram_phi[prev_tag][tag] += 1
            bigram_phi[prev_pred][prediction] -= 1
            prev_tag = tag
            prev_pred = prediction
        weights[:self.num_classes * IMG_SIZE] += pixels_phi
        weights[self.num_classes * IMG_SIZE:] += np.ravel(bigram_phi)
        return weights


def perceptron(x, method):
    weights = method.initialize_weights()
    total = len(x)
    final_weights = []
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch #{epoch}')
        weights_adj = [weights]
        x = shuffle(x, random_state=epoch)
        for index, word in enumerate(x):
            letters, letter_tags = word
            if index % 500 == 1:
                print(f'-- {round(float(index) / total * 100, 2)}%')
                test(pred_x[:50], np.average(weights_adj, 0), method)
            predictions = method.predict(weights, letters)
            tag_vec = [classes[tag] for tag in letter_tags]
            weights = method.update(weights, letters, tag_vec, predictions)
            weights_adj.append(weights)
        final_weights.append(np.average(weights_adj, 0))
        test(pred_x, np.average(final_weights, 0), method)
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


def test_predict(test_set, weights, method, inv_tags):
    all_predictions = []
    for word in test_set:
        predictions = method.predict(weights, word[0])
        all_predictions.extend([inv_tags[prediction] for prediction in predictions])
    with open('test_' + method_type + '.pred', 'w') as file:
        file.write('\n'.join(all_predictions))


if __name__ == '__main__':
    method_type = 'c'
    method = Bigram_structured_perceptron(NUM_CLASSES)

    if 'train' in sys.argv:
        data, tags = load_data(TRAIN)
        np.random.shuffle(data)
        cutoff = round(len(data) * 0.8)
        classes = {v: i for i, v in enumerate(set(tags).union(START))}
        train_x = data[:cutoff]
        pred_x = data[cutoff:]
        model = perceptron(train_x, method)
        pickle.dump([model, classes], open('model_' + method_type, 'wb'))

    if 'test' in sys.argv:
        test_data, _ = load_data(TEST)
        model, classes = pickle.load(open('model_' + method_type, 'rb'))
        test_predict(test_data, model, method, {v: k for k, v in classes.items()})

    if 'train' not in sys.argv and 'test' not in sys.argv:
        raise Exception("Illegal command line parameters received!")



