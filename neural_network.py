import csv, gzip, random, sys, atexit, time, matplotlib.pyplot as plt, numpy as np, numpy.matlib
from multiprocessing import cpu_count, Pool, Process
import activations
from losses import quadratic, cross_entropy


"""
cmd : python neural_network.py 784 30 10 TrainDigitX.csv.gz TrainDigitY.csv.gz TestDigitX.csv.gz PredictDigitY.csv.gz

cmd : python neural_network.py 784 30 10 TestDigitX.csv.gz TestDigitY.csv.gz TrainDigitX.csv.gz PredictDigitY.csv.gz
"""

class MultilayerPerceptron:
    """
    A neural network with fully connected layers.
    """
    
    def __init__(self, layers, epochs, batchsize, learningrate):
        self.layers = layers
        self.epochs = epochs
        self.batchsize = batchsize
        self.learningrate = learningrate
        self.inference = False
        self.savePrediction = False
        self.count = 0
        self.outGradientAvg = np.zeros((10, 30), dtype=float)
        self.hiddenGradientAvg = np.zeros((784, 30), dtype=float)
        self.biasOutGradientAvg = np.zeros((10),dtype=float)
        self.biasHiddenGradientAvg = np.zeros((10),dtype=float)
        self.predictionList = []
        self.main()

    # key : 'weight' or 'bias'
    def data(self, trainset, trainset_label):
        """
        Reads in the data from files specified by command line arguments
        """

        print("\nReading Data...")

        # Trainset : 'TrainDigitX.csv.gz'
        # Trainset_label : 'TrainDigitY.csv.gz'
        with gzip.open(trainset, 'rt') as csvfile:
            input_data = list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))                
        print("\n", trainset, "100% -", len(input_data), ": instances")

        with gzip.open(trainset_label, 'rt') as csvfile:
            input_label = list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
        print("\n", trainset_label, "100% -", len(input_label), ": instances")

        return input_data, input_label

    def variable(self, i, key):
        if key == 'weight':
            return np.array([[random.uniform(-1, 1) for i in range(self.layers[i])] for j in range(self.layers[i-1])])
        elif key == 'bias':
            return np.array([random.uniform(-1, 1) for i in range(self.layers[i])])            

    def onehotencoding(self, input_label_batch):
        """
        Converts the label to one hot encoding
        return: one hot array of label
        """
        encodedLabel = []
        for k in range(len(input_label_batch)):
            label = int(input_label_batch[k][0])
            temp = [0 for j in range(10)]
            temp[label] = 1
            encodedLabel.append(temp)

        return encodedLabel

    def feedforward(self, inputs):
        """
        Feeds the input through the network layers to the softmax function
        """
        # Hidden Layer 1 
        z1Sum = np.matmul(self.weights['hidden1'].T, inputs) + self.bias['hidden1']
        z1 = activations.sigmoid(z1Sum, 'normal')
        
        # Hidden Layer 2
        #z2 = activations.sigmoid((np.einsum('ij, j->i', self.weights['hidden2'], z1) + self.bias['hidden2']), 'normal')

        # Output Layer
        outsum = np.matmul( self.weights['out'].T, z1) + self.bias['out']
        
        prediction = activations.softmax(outsum, 'normal')
        #self.predictionList.append(np.argmax(prediction))
        #print(prediction)
        
        
        return prediction, outsum, z1, z1Sum

    def backpropagate(self, prediction, encodedLabels, outsum, z1, z1Sum, x):
        '''
        Applies the chain rule to backpropagate the network
        '''
        z1 = z1.reshape(self.layers[1], 1)
        z1 = z1.T
        x = np.array(x)
        x = x.reshape(self.layers[0], 1)
        outDelta = np.multiply(quadratic(prediction, encodedLabels, 'derivative'), (prediction*(1 - prediction)))
        
        outDelta = outDelta.reshape(self.layers[-1], 1)
        outGradient = self.learningrate*np.matmul(outDelta, z1)
        z1 = z1.T
        hiddenDelta = np.matmul(self.weights['out'], outDelta)*(z1*(1-z1))
        hiddenGradient = self.learningrate*np.matmul(x,hiddenDelta.T)

        return outGradient, hiddenGradient

    def updateWeights(self, outGradient, hiddenGradient):
        outGradient = outGradient.T
        #hiddenGradient = hiddenGradient.T
        self.weights['out'] -= outGradient
        self.weights['hidden1'] -= hiddenGradient
        

        

    def minibatch(self, input_data_batch, input_label_batch):
        """
        Controls the process of feeding through and backpropagating the network by passing individual examples from the minibatch
        Calculates the error for each batch
        """
        cores = cpu_count()
        encodedLabels = self.onehotencoding(input_label_batch)
        i = 0
        
        errorAvg = 0
        gradientAvg = 0
        processes = []
        
        '''for row in input_data_batch:
            p = Process(target=self.feedforward, args=(row,))
            processes.append(p)
            pool = Pool(processes=cores)
        [x.start() for x in processes]'''

        '''if __name__ == '__main__':
            pool = Pool(processes=cpuCount)
            parallelBatch = pool.apply_async(self.feedforward, (input_data_batch,))
            prediction, outsum, z1, z1Sum = parallelBatch.get(timeout=1)'''
        #gradientAvg = np.zeros()
        self.outGradientAvg = np.zeros((self.layers[-1], self.layers[1]), dtype=float)
        self.hiddenGradientAvg = np.zeros((self.layers[0], self.layers[1]), dtype=float)
        self.biasOutGradientAvg = np.zeros((1,10),dtype=float)
        self.biasHiddenGradientAvg = np.zeros((30,),dtype=float)
        for row in input_data_batch:
            row = np.array(row)
            '''pool = Pool(processes=cpuCount)
            parallelBatch = pool.apply_async(self.feedforward, (input_data_batch,))
            prediction, outsum, z1, z1Sum = parallelBatch.get(timeout=1)'''
            prediction, outsum, z1, z1Sum = self.feedforward(row)
            errorAvg += quadratic(prediction, encodedLabels[i], 'normal')
            
            if self.inference == False:
                outGradient, hiddenGradient = self.backpropagate(prediction, encodedLabels[i], outsum, z1, z1Sum, row)
                self.outGradientAvg += outGradient
                self.hiddenGradientAvg += hiddenGradient
                #self.updateWeights(self.outGradientAvg/self.batchsize, self.hiddenGradientAvg/self.batchsize)
            else:
                if np.argmax(prediction) == np.argmax(encodedLabels[i]):
                    #print(np.argmax(prediction), np.argmax(encodedLabels[i]))
                    
                    self.count += 1
                else:
                    pass
                    #print(np.argmax(prediction), np.argmax(encodedLabels[i]))
                #print("\n",prediction, encodedLabels[i])
            i += 1
        
        self.updateWeights(self.outGradientAvg/len(input_data_batch), self.hiddenGradientAvg/len(input_data_batch))
        errorAvg = errorAvg / len(input_data_batch)
        print(" Error:", errorAvg, end='', flush=True)
        return errorAvg

    def train(self, weights, bias, input_data, input_label):
        """
        Trains the network by iterating through all minibatches and through all epochs
        Randomly shuffles the data and splits it into minibatches
        """

        #print("\n-------Training-------\n\nEpochs:", self.epochs, "- Batch Size:", self.batchsize, "- Learning Rate (\u03B7):", self.learningrate)
        error = []
        labeledData = list(zip(input_data, input_label))
        if self.savePrediction == False:
            np.random.shuffle(labeledData)

        input_data, input_label = zip(*labeledData)

        # seperate data into batches
        input_data_batch = [input_data[x: x + self.batchsize] for x in np.arange(0, len(input_data), self.batchsize)]
        input_label_batch = [input_label[x : x + self.batchsize] for x in np.arange(0, len(input_label), self.batchsize)]

        i = 0
        
        for row in input_data_batch:
            print("\rBatch:", i + 1, end=' -', flush=True)
            error.append(self.minibatch(row, input_label_batch[i]))
            i += 1
        #plt.plot(error, color='black')
        #plt.show()
    def main(self):
        
        trainset = str(sys.argv[4])
        trainset_label = str(sys.argv[5])
        testset = str(sys.argv[6])
        testlabel = str(sys.argv[8])
        testset_predict = str(sys.argv[7])
        
        self.weights = {
            'hidden1' : self.variable(1, 'weight'),
            #'hidden2' : self.variable(2, 'weight'),
            'out' : self.variable(len(layers) - 1, 'weight')
        }
        self.bias = {
            'hidden1' : self.variable(1, 'bias'),
            #'hidden2' : self.variable(2, 'bias'),
            'out' : self.variable(len(layers) - 1, 'bias')
        }
        print("\nInput:", self.layers[0], "Hidden:", self.layers[1], "Output:", self.layers[-1])
        input_data_train, input_label_train = self.data(trainset, trainset_label)
        input_data_test, input_label_test = self.data(testset, testlabel)
        accuracy = []
        for epoch in range(self.epochs):
                
            
            self.count = 0
            print("\n-------Training-------")
            print("\nEpoch:", epoch + 1)
            self.train(self.weights, self.bias, input_data_train, input_label_train)
           
            print("\n\n-------Testing-------\n")            
            self.inference = True
            self.train(self.weights, self.bias, input_data_test, input_label_test)
            print("\nAccuracy", self.count/len(input_label_test))
            accuracy.append(self.count/len(input_data_test))
            self.inference = False
        self.inference = True
        self.savePrediction = True
        input_data_test, input_label_test = self.data(testset, testlabel)
        #self.train(self.weights, self.bias, input_data_test, input_label_test)
        for i in range(len(input_data_test)):
            prediction, x, y, z = self.feedforward(input_data_test[i])
            self.predictionList.append(np.argmax(prediction))

        if self.savePrediction == True:
            with open('PredictDigitY.csv', 'w+') as file:
                writer = csv.writer(file, lineterminator='\n')
                for row in self.predictionList:
                    writer.writerow([row])
            
        xaxis = [i for i in range(1, len(accuracy) + 1)]
        plt.plot(xaxis, accuracy, color='black')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Epochs:30 - Batch Size: 10- Learning Rate:3')
        '''with open('TextData/quadBatch1.txt', 'a+') as file:
            for row in accuracy:
                file.write(str(np.max(accuracy)) )
                file.write('\n')'''
        #file.flush()
        #file.close()
        #plt.show()



if __name__ == '__main__':
    # network Parameters
    epochs = 30
    batchsize = 20
    learningrate = 3
    n_input = int(sys.argv[1])
    n_hidden_1 = int(sys.argv[2])
    n_hidden_2 = 30
    n_output = int(sys.argv[3])
    layers = [n_input, n_hidden_1, n_output]

    mlp = MultilayerPerceptron(layers, epochs, batchsize, learningrate)
