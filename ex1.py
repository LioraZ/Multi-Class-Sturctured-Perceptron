from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import numpy as np
from collections import defaultdict
# read data
mnist = fetch_mldata("MNIST original", data_home="./data")
x_test = np.loadtxt('x4pred.txt')

eta = 0.1
lam = 0.01
NUM_CLASSES = 4
EPOCHS = 1

def svm(x, y, translation):
    weights = [np.zeros([len(x[0])])]
    x, y = shuffle(x, y, random_state=1)
    adj_y = [translation[int(val)] for val in y]
    for epoch, (ex, classname) in enumerate(zip(x, adj_y)):
        etaT = eta / ((epoch + 1) ** 0.5)
        classify = (1 - classname * weights[-1].dot(ex))
        if classify >= 0:
            app = weights[-1] * (1 - (etaT * lam)) + (etaT * classname * ex)
        else:
            app = weights[-1] * (1 - (etaT * lam))
        weights.append(app)
        #if epoch > 10000:
        #    break
    return np.sum(weights, axis=0)

def get_one_vs_all_matrix(NUM_CLASSES):
    one_vs_all_dim = [NUM_CLASSES, NUM_CLASSES]
    matrix = -1 * np.ones(one_vs_all_dim) + 2 * np.eye(NUM_CLASSES)
    return {'name': 'onevall', 'matrix': matrix}

def get_all_pairs_matrix(NUM_CLASSES):
    all_pairs = []
    seen = defaultdict(lambda: defaultdict(dict))
    for i in range(NUM_CLASSES):
        for j in range(i, NUM_CLASSES):
            if (not seen[i].get(j) and i != j):
                column = np.zeros([NUM_CLASSES])
                column[i] = 1
                column[j] = -1
                all_pairs.append(column)
            seen[i][j] = True
    matrix = np.array(all_pairs).transpose()
    return {'name': 'allpairs', 'matrix': matrix}

def get_random_matrix(NUM_CLASSES):
    matrix = np.random.randint(low=-1, high=1, size=[NUM_CLASSES, NUM_CLASSES])
    return {'name': 'randm', 'matrix': matrix}
    
def load_set(set_type):
    X, Y = [], []
    if set_type == 'train':
        X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    else:
        X, Y = mnist.data[60000:] / 255., mnist.target[60000:]
    x = [ex for ex, ey in zip(X, Y) if ey in range(NUM_CLASSES)]
    y = [ey for ey in Y if ey in range(NUM_CLASSES)]
    return x, y

def hamm_diff_func(matrix, raw_result):
    hamm_dif = []
    matrix = matrix
    raw_result = np.sign(raw_result)
    for row in matrix:
        hamm_dif.append(np.sum((1 - np.sign(row * raw_result)) / 2))
    return (hamm_dif, 'ham')

def loss_diff_func(matrix, raw_result):
    loss_diff = []
    matrix = matrix
    for row in matrix:
        calc = np.maximum(1 - (row * raw_result), np.zeros(len(row)))
        loss_diff.append(np.sum(calc))
    return (loss_diff, 'loss')
    
        
    


if __name__ == '__main__':
    x, y = load_set('train')
    
    x, y = shuffle(x, y, random_state=1)
                
    all_pairs = get_all_pairs_matrix(NUM_CLASSES)
    one_vs_all = get_one_vs_all_matrix(NUM_CLASSES)
    random_mat = get_random_matrix(NUM_CLASSES)
    
    for method in [one_vs_all, all_pairs, random_mat]:
        models = []
        #Train
        for row in method['matrix'].transpose():
            models.append(svm(x, y, row))
        
        #Evaluate
        for set_type in ['train', 'pred']:
            x, y = load_set(set_type)
            total = 0.0
            tally = defaultdict(lambda: defaultdict(float))
            for example, correct_class in zip(x, y):
                f = []
                for mod in models:
                    f.append(mod.dot(example))
                for loss_f in [hamm_diff_func, loss_diff_func]:
                    loss_diff, name = loss_f(method['matrix'], f)
                    if np.argmin(loss_diff) != int(correct_class):
                        tally[name]['wrong'] += 1.0
                    else:
                        tally[name]['right'] += 1.0
                total += 1.0
            for key in tally.keys():
                perc = round(tally[key]['right'] / total * 100, 3)
                print('After using model ' + method['name'] + ' and set type ' + set_type)
                print('Got ' + str(perc) + '% correct (' + key + ')')
                print('')
                
        #Test
        predictions = defaultdict(list)
        for example in x_test:
            f = []
            for mod in models:
                f.append(mod.dot(example))
            for loss_f in [hamm_diff_func, loss_diff_func]:
                loss_diff, name = loss_f(method['matrix'], f)
                predictions[name].append(np.argmin(loss_diff))
        for key in predictions.keys():
            mtdn = method['name']
            filename = f'test.{mtdn}.{key}.pred'
            np.savetxt(filename, predictions[key])
    