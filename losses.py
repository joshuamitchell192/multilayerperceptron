import numpy as np, math
import activations

def quadratic(prediction, label, key):
    if key == 'normal':
        #print(prediction, label)
        return np.sum(0.5*(label - prediction)**2)
    elif key == 'derivative':
        return prediction - label
    #elif key == 'delta':
    #    return np.einsum('ij, j->i', (prediction - label), activations.sigmoid(inputs, 'derivative'))
    else:
        print('Incorrect key: use either normal or derivative')

def cross_entropy(prediction, label, key):
    if key == 'normal':
        #print(label, prediction, np.log(prediction))
        #print(np.sum(prediction))
        #return (np.multiply(label, np.log(prediction)) + np.multiply(1-label), np.log(1-prediction))
        return (np.argmax(label) * math.log(np.max(prediction)) + (1 - np.argmax(label)) * math.log(1 - np.max(prediction)))
    elif key == 'derivative':
        return prediction - label
    else:
        print('Incorrect key: use either normal or derivative')
    