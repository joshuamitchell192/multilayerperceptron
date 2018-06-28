import numpy as np
    
def sigmoid(vectors, key):
    if key == 'normal':
        return 1 / (1 + np.exp(-vectors))
    elif key == 'derivative':
        return sigmoid(vectors, 'normal') * (1 / sigmoid(vectors, 'normal'))
    else:
        print('Incorrect key: use either normal or derivative')

def relu(vectors, key):
    if key == 'normal':
        vectors[vectors <= 0] = 0
        return vectors
    elif key == 'derivative':
        vectors[vectors > 0] = 1
        vectors[vectors <= 0] = 0
        return vectors
    else:
        print('Incorrect key: use either normal or derivative')

def leakyRelu(x, key):
    if key == 'normal':
        if x > 0:
            return x
        else:
            return 0.01*x
    elif key == 'derivative':
        if x > 0:
            return 1
        else:
            return 0.01
    else:
        print('Incorrect key: use either normal or derivative')

def elu(x, a, key):
    if key == 'normal':
        if x >= 0:
            return x
        else:
            return a*(np.exp(x) - 1)
    elif key == 'derivative':
        if x >= 0:
            return 1
        else:
            return a*(np.exp(x))
    else:
        print('Incorrect key: use either normal or derivative')

def celu(x, a, key):
    if key == 'normal':
        if x >= 0:
            return x
        else:
            return a*(np.exp(x/a) - 1)
    elif key == 'derivative':
        if x >= 0:
            return 1
        else:
            return np.exp(x/a)
    else:
        print('Incorrect key: use either normal or derivative')
    
def softmax(x, key):
    if key == 'normal':
        #print('\n', len(x), x)
        e_x = np.exp(x - np.max(x))
        #print(e_x, '\n', e_x.sum())
        out = e_x / np.sum(e_x)
        #print(out)
        return out
        
    else:
        print('Incorrect key: use either normal or derivative')
    



