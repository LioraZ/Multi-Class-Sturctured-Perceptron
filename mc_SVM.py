from sklearn.utils import shuffle
from sklearn.svm import SVC
import numpy as np
import pickle

EPOCHS = 3
LR = 0.01
DATA_DIR = 'data'
TRAIN = '/letters.train.data'
TEST = '/letters.test.data'
START = '<s>'

ID = 0
LETTER = 1
NEXT_ID = 2
FOLD = 3
PIXELS = 4

IMG_SIZE = 128 #8 * 16


"""def mc_SVM():
    weights = [np.random.rand(len(classes), IMG_SIZE * len(classes))]
    for epoch in range(1, EPOCHS + 1):
        print("Epoch #" + str(epoch) + ": ")
        eta = 0.01
        data = shuffle(train_data, random_state=1)
        good = bad = 0.0
        for x, y in zip(data[0], data[1]):
            x = x[PIXELS]
            pad = np.zeros(IMG_SIZE * len(x))
            pad[classes[y]:classes[y] + len(classes)] = x
            x = pad
            etaT = eta / (epoch ** 0.5)
            W = weights[-1]
            prediction = np.argmax(np.dot(W, x))
            if prediction != classes[y]:
                next_W = update_W(x, classes[y], prediction, W, etaT)
                weights.append(next_W)
                bad += 1
            else:
                good += 1
        print("Accuracy on train: " + str(100 * (good / (good + bad))) + "%")
        #weights = [np.sum(weights, axis=0)]
        weights = [weights[-1]]
        good = bad = 0.0
        for x, y in zip(dev_data[0], dev_data[1]):
            x = x[PIXELS]
            W = weights[-1]
            prediction = np.argmax(np.dot(W, x))
            good += 1 if prediction == classes[y] else 0
            bad += 1 if prediction != classes[y] else 0
        print("Accuracy on dev: " + str(100 * (good / (good + bad))) + "%\n")
    return weights[-1]"""


def structured_SVM():
    weights = [np.random.rand(len(classes), IMG_SIZE)]
    for epoch in range(1, EPOCHS + 1):
        print("Epoch #" + str(epoch) + ": ")
        eta = 0.01
        data = shuffle(train_data, random_state=1)
        good = bad = 0.0
        for x, y in zip(data[0], data[1]):
            x = x[PIXELS]
            etaT = eta / (epoch ** 0.5)
            W = weights[-1]
            prediction = np.argmax(np.dot(W, x))
            if prediction != classes[y]:
                next_W = update_W(x, classes[y], prediction, W, etaT)
                weights.append(next_W)
                bad += 1
            else:
                good += 1
        print("Accuracy on train: " + str(100 * (good / (good + bad))) + "%")
        # weights = [np.sum(weights, axis=0)]
        weights = [weights[-1]]
        good = bad = 0.0
        for x, y in zip(dev_data[0], dev_data[1]):
            x = x[PIXELS]
            W = weights[-1]
            prediction = np.argmax(np.dot(W, x))
            good += 1 if prediction == classes[y] else 0
            bad += 1 if prediction != classes[y] else 0
        print("Accuracy on dev: " + str(100 * (good / (good + bad))) + "%\n")
    return weights[-1]

def get_W_structured_SVM():
    return np.zeros(len(classes), IMG_SIZE * len(classes))

def pad_x_structured_SVM(x, label):

    def pad_x(x, y):
        x = x[PIXELS]
        pad = np.zeros(IMG_SIZE * len(x))
        pad[classes[y]:classes[y] + len(classes)] = x
        return pad

    if label in classes.values():
        return pad_x(x, label)
    return [pad_x(x, v) for _, v in classes.items()]



def get_W_mc_SVM():
    return np.zeros((len(classes), IMG_SIZE))

def pad_x_mc_SVM(x, label):
    return x

def update_W_mc_SVM(x, label, prediction, W, eta):
    next_W = W.copy()
    for i in range(len(classes)):
        if i == label:
            next_W[i] = (W[i] + x) * LR
        elif i == prediction:
            next_W[i] = (W[i] - x) * LR
    return next_W


def learn(get_W, pad_x, update_W):
    W = get_W()
    for epoch in range(1, EPOCHS + 1):
        print("Epoch #" + str(epoch) + ": ")
        W = train(W, pad_x, update_W)
        dev(W, pad_x)
    predictions = test(W, pad_x)
    write_predictions(predictions)


def train(W, pad_x, update_W):
    weights = [W]
    eta = 0.01
    data = shuffle(train_data, random_state=1)
    good = bad = 0.0
    for x, y in zip(data[0], data[1]):
        x = pad_x(x[PIXELS], y)
        etaT = 1
        #etaT = eta / (epoch ** 0.5)
        W = weights[-1]
        prediction = np.argmax(np.dot(W, x))
        if prediction != classes[y]:
            next_W = update_W(x, classes[y], prediction, W, etaT)
            weights.append(next_W)
            bad += 1
        else:
            good += 1
    print("Accuracy on train: " + str(100 * (good / (good + bad))) + "%")
    return np.sum(weights, axis=0) / bad


def dev(W, pad_x):
    good = bad = 0.0
    for x, y in zip(dev_data[0], dev_data[1]):
        x = pad_x(x[PIXELS], None)
        prediction = np.argmax(np.dot(W, x))
        good += 1 if prediction == classes[y] else 0
        bad += 1 if prediction != classes[y] else 0
    print("Accuracy on dev: " + str(100 * (good / (good + bad))) + "%\n")


def test(W, pad_x):
    inv_classes = {v: k for k, v in classes.items()}
    predictions = []
    for x in test_data[0]:
        x = pad_x(x[PIXELS], None)
        predictions.append(inv_classes[np.argmax(np.dot(W, x))])
    return predictions


def write_predictions(predictions):
    with open('test.pred', 'w') as file:
        file.write('\n'.join(predictions))


def load_data(data_path):
    data = []
    tags = []
    with open(DATA_DIR + data_path, 'r') as file:
        for line in file:
            fields = line.split()
            pixels = np.array([int(x) for x in fields[6:]])
            item = [int(fields[0]), fields[1], int(fields[2]), int(fields[3]), pixels]
            data.append(item)
            tags.append(item[LETTER])
    return [data, tags]


def svm():
    eta = 0.01
    good = bad = 0.0
    weights = [np.zeros((len(classes), IMG_SIZE))]
    #x, y = shuffle(x, y, random_state=1)
    #adj_y = [translation[int(val)] for val in y]
    for i, (x, y) in enumerate(zip(train_data[0], train_data[1])):
        etaT = eta / ((i + 1) ** 0.5)
        x = x[PIXELS]
        classify = np.argmax(np.dot(weights[-1], x))
        real_y = np.zeros((len(classes)))
        real_y[classes[y]] = 1
        real_y[classify] = -1
        update_W = np.outer(real_y, x)
        weights.append(update_W + etaT * weights[-1])
        good += 1 if classify == classes[y] else 0
        bad += 1 if classify != classes[y] else 0
        """if classify >= 0:
            app = weights[-1] * (1 - (etaT * lam)) + (etaT * classname * ex)
        else:
            app = weights[-1] * (1 - (etaT * lam))
        weights.append(app)"""
        #if epoch > 10000:
        #    break
    print(str(100 *(good/ (good + bad))))
    return np.sum(weights, axis=0)


def mc_svm():
    X = [x[PIXELS] for x in train_data[0]]
    Y = [classes[y] for y in train_data[1]]
    svm = SVC()
    svm.fit(X, Y)
    predictions = svm.predict([x[PIXELS] for x in dev_data[0]])
    count = np.count_nonzero(predictions - [classes[y] for y in dev_data[1]])
    print(str(count/ len(predictions)))
    return svm

if __name__ == '__main__':
    train_data = load_data(TRAIN)
    classes = {v: i for i, v in enumerate(set(train_data[1]).union(START))}
    div = int(len(train_data[0]) * 0.8)
    dev_data = [train_data[0][div:], train_data[1][div:]]
    train_data = [train_data[0][:div], train_data[1][:div]]
    test_data = load_data(TEST)
    model = mc_svm()
    pickle.dump(model, open('multi_class_svm.save', 'wb'))
    #W = svm()
    #W = mc_SVM()
    #learn(get_W_mc_SVM, pad_x_mc_SVM, update_W_mc_SVM)
