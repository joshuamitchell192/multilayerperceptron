import csv
import gzip
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from multiprocessing import cpu_count

import activations
from losses import quadratic


# cmd : python neural_network.py 784 30 10 TrainDigitX.csv.gz TrainDigitY.csv.gz TestDigitX.csv.gz TestDigitY.csv.gz PredictDigitY.csv.gz

# cmd : python neural_network.py 784 30 10 TestDigitX.csv.gz TestDigitY.csv.gz TrainDigitX.csv.gz TrainDigitY.csv.gz PredictDigitY.csv.gz


class MultilayerPerceptron:
    """
    A neural network with fully connected layers.
    """
    
    def __init__(self, layers, epochs, batchsize, learningrate):
        self.layers = layers
        self.epochs = epochs
        self.batchsize = batchsize
        self.learningrate = learningrate
        self.weights = {}
        self.bias = {}
        self.inference = False
        self.savePrediction = False
        self.load_saved_weights = True
        self.count = 0
        self.outGradientAvg = np.zeros((10, 30), dtype=np.float16)
        self.hiddenGradientAvg = np.zeros((784, 30), dtype=np.float16)
        self.biasOutGradientAvg = np.zeros(10, dtype=np.float16)
        self.biasHiddenGradientAvg = np.zeros(10, dtype=np.float16)
        self.predictionList = []

    @staticmethod
    def data(trainset, trainset_label):
        """
        Reads in the data from files specified by command line arguments

        :returns
            input_data - The data for input to the network
            input_label - The data's labels
        """

        print("\nReading Data...")

        # Train_set : 'TrainDigitX.csv.gz'
        # Train_set_label : 'TrainDigitY.csv.gz'
        with gzip.open(trainset, 'rt') as csvfile:
            input_data = list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))                
        print("\n", trainset, "100% -", len(input_data), ": instances")

        with gzip.open(trainset_label, 'rt') as csvfile:
            input_label = list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
        print("\n", trainset_label, "100% -", len(input_label), ": instances")

        return input_data, input_label

    def variable(self, i, key):
        """
        Creates and returns matrices of weight and bias values
        key values - 'weight' or 'bias'
        """
        if key == 'weight':
            return np.array([[random.uniform(-1, 1) for i in range(self.layers[i])] for j in range(self.layers[i-1])], dtype=np.float16)
        elif key == 'bias':
            return np.array([random.uniform(-1, 1) for i in range(self.layers[i])], dtype=np.float16)
        else:
            print("Incorrect key value passed to variable function. No matrix was returned.")

    @staticmethod
    def one_hot_encoding(input_label_batch):
        """
        Converts the label to one hot encoding
        return: one hot array of label
        """
        encoded_label = []
        for k in range(len(input_label_batch)):
            label = int(input_label_batch[k][0])
            temp = [0 for j in range(10)]
            temp[label] = 1
            encoded_label.append(temp)

        return encoded_label

    def feedforward(self, inputs):
        """
        Feeds the input through the network layers to the softmax function
        """
        # Hidden Layer 1 
        z1_sum = np.matmul(self.weights['hidden1'].T, inputs) + self.bias['hidden1']
        z1 = activations.sigmoid(z1_sum, 'normal')
        
        # Hidden Layer 2
        # z2 = activations.sigmoid((np.einsum('ij, j->i', self.weights['hidden2'], z1) + self.bias['hidden2']), 'normal')

        # Output Layer
        out_sum = np.matmul(self.weights['out'].T, z1) + self.bias['out']
        
        prediction = activations.softmax(out_sum, 'normal')

        return prediction, out_sum, z1, z1_sum

    def backpropagate(self, prediction, encoded_labels, h1, x):
        """
        Applies the chain rule to backpropagate the network
        """
        h1 = h1.reshape(self.layers[1], 1)
        h1 = h1.T
        x = np.array(x)
        x = x.reshape(self.layers[0], 1)
        out_delta = np.multiply(quadratic(prediction, encoded_labels, 'derivative'), (prediction * (1 - prediction)))
        
        out_delta = out_delta.reshape(self.layers[-1], 1)
        out_gradient = self.learningrate*np.matmul(out_delta, h1)
        h1 = h1.T
        hidden_delta = np.matmul(self.weights['out'], out_delta)*(h1 * (1 - h1))
        hidden_gradient = self.learningrate*np.matmul(x, hidden_delta.T)

        return out_gradient, hidden_gradient

    def update_weights(self, out_gradient, hidden_gradient):

        out_gradient = out_gradient.T
        self.weights['out'] -= out_gradient
        self.weights['hidden1'] -= hidden_gradient

    def mini_batch(self, input_data_batch, input_label_batch):
        """
        Controls the process of feeding through and
        backpropagating the network by passing individual examples from the mini_batch.

        Calculates the error for each batch
        """
        cores = cpu_count()
        encoded_labels = self.one_hot_encoding(input_label_batch)
        i = 0
        
        error_avg = 0
        processes = []
        
        '''for row in input_data_batch:
            p = Process(target=self.feedforward, args=(row,))
            processes.append(p)
            pool = Pool(processes=cores)
        [x.start() for x in processes]'''

        '''if __name__ == '__main__':
            pool = Pool(processes=cpuCount)
            parallelBatch = pool.apply_async(self.feedforward, (input_data_batch,))
            prediction, outsum, h1, h1_sum = parallelBatch.get(timeout=1)'''

        self.outGradientAvg = np.zeros((self.layers[-1], self.layers[1]), dtype=np.float16)
        self.hiddenGradientAvg = np.zeros((self.layers[0], self.layers[1]), dtype=np.float16)
        self.biasOutGradientAvg = np.zeros((1, 10), dtype=np.float16)
        self.biasHiddenGradientAvg = np.zeros((30,), dtype=np.float16)

        for row in input_data_batch:
            row = np.array(row)
            '''pool = Pool(processes=cpuCount)
            parallelBatch = pool.apply_async(self.feedforward, (input_data_batch,))
            prediction, outsum, h1, h1_sum = parallelBatch.get(timeout=1)'''
            prediction, outsum, h1, h1_sum = self.feedforward(row)
            error_avg += quadratic(prediction, encoded_labels[i], 'normal')
            
            if not self.inference:
                out_gradient, hidden_gradient = self.backpropagate(prediction, encoded_labels[i], h1, row)
                self.outGradientAvg += out_gradient
                self.hiddenGradientAvg += hidden_gradient
                # self.updateWeights(self.outGradientAvg/self.batchsize, self.hiddenGradientAvg/self.batchsize)
            else:
                if np.argmax(prediction) == np.argmax(encoded_labels[i]):
                    
                    self.count += 1
                else:
                    pass
            i += 1
        
        self.update_weights(self.outGradientAvg/len(input_data_batch), self.hiddenGradientAvg/len(input_data_batch))
        error_avg = error_avg / len(input_data_batch)
        # print(" Error:", error_avg, end='', flush=True)
        return error_avg

    def train(self, input_data, input_label):
        """
        Trains the network by iterating through all minibatches and through all epochs
        Randomly shuffles the data and splits it into minibatches
        """

        # print("\n-------Training-------\n\nEpochs:", self.epochs, "- Batch Size:", self.batchsize, "- Learning Rate (\u03B7):", self.learningrate)
        error = []
        labeled_data = list(zip(input_data, input_label))
        if not self.savePrediction:
            np.random.shuffle(labeled_data)

        input_data, input_label = zip(*labeled_data)

        # seperate data into batches
        input_data_batch = [input_data[x: x + self.batchsize] for x in np.arange(0, len(input_data), self.batchsize)]
        input_label_batch = [input_label[x: x + self.batchsize] for x in np.arange(0, len(input_label), self.batchsize)]

        i = 0
        
        for row in input_data_batch:
            print("\rBatch:", i + 1, end=' -', flush=True)
            error.append(self.mini_batch(row, input_label_batch[i]))
            i += 1
        # plt.plot(error, color='black')
        # plt.show()

    def run(self):
        
        train_set = str(sys.argv[4])
        train_set_label = str(sys.argv[5])
        test_set = str(sys.argv[6])
        test_label = str(sys.argv[7])
        # testset_predict = str(sys.argv[8])
        
        if (self.load_saved_weights):
            self.weights = self.load_weights("weights")
            self.bias = self.load_weights("bias")
        else:
            self.weights = {
                'hidden1': self.variable(1, 'weight'),
                # 'hidden2' : self.variable(2, 'weight'),
                'out': self.variable(len(layers) - 1, 'weight')
            }
            self.bias = {
                'hidden1': self.variable(1, 'bias'),
                # 'hidden2' : self.variable(2, 'bias'),
                'out': self.variable(len(layers) - 1, 'bias')
            }

        print("\nInput:", self.layers[0], "\nHidden 1:", self.layers[1], "\nOutput:", self.layers[-1])
        input_data_train, input_label_train = self.data(train_set, train_set_label)
        input_data_test, input_label_test = self.data(test_set, test_label)
        accuracy = []

        for epoch in range(self.epochs):

            self.count = 0
            print("\n-------Training-------")
            print("\nEpoch:", epoch + 1)
            self.train(input_data_train, input_label_train)
           
            print("\n\n-------Testing-------\n")            
            self.inference = True
            self.train(input_data_test, input_label_test)

            print("\nAccuracy", self.count/len(input_label_test))
            accuracy.append(self.count/len(input_data_test))
            self.inference = False

        self.save_weights(self.weights, self.bias)
            
        xaxis = [i for i in range(1, len(accuracy) + 1)]
        plt.plot(xaxis, accuracy, color='black')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Epochs:30 - Batch Size: 10- Learning Rate:3')
        plt.show()

    def load_weights(self, type):
        if type == "weights":


            with open("./saved_model/weights.npy", 'rb+') as file:

                hidden1 = np.load(file, allow_pickle=True)
                print(hidden1)
                out = np.load(file, allow_pickle=True)
                return {
                    'hidden1': hidden1,
                    # 'hidden2' : self.variable(2, 'weight'),
                    'out': out
                }

        if type == "bias":
            with open("./saved_model/bias.npy", 'rb+') as file:
                hidden1 = np.load(file, allow_pickle=True)
                out = np.load(file, allow_pickle=True)
                return {
                    'hidden1': hidden1,
                    # 'hidden2' : self.variable(2, 'weight'),
                    'out': out
                }

    def save_weights(self, weights, bias):
        
        with open("./saved_model/weights.npy", 'wb+') as file:
            np.save(file, weights['hidden1'])
            np.save(file, weights['out'])

        with open("./saved_model/bias.npy", 'wb+') as file:
            np.save(file, bias['hidden1'])
            np.save(file, bias['out'])

if __name__ == '__main__':

    # network Parameters
    epochs = 20
    batchsize = 20
    learningrate = 3
    n_input = int(sys.argv[1])
    n_hidden_1 = int(sys.argv[2])
    n_hidden_2 = 30
    n_output = int(sys.argv[3])
    layers = [n_input, n_hidden_1, n_output]

    mlp = MultilayerPerceptron(layers, epochs, batchsize, learningrate)
    mlp.run()
