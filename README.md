# multilayer perceptron
## Overview
An implementation of a multilayer perceptron with numpy. It includes the MNIST Training and Testing files, as well as a file to write the networks predictions to. 
The network trains on 50 000 examples of handwritten digits and is tested on 10 000 examples to get the networks accuracy. By default the network has 784 inputs, 30 hidden neurons and 10 outputs. The learning rate is set to 3, and mini batch size is 20.

![alt text](mlprun.PNG "Example")

## Dependancies
You will need to be using python 3 and have numpy and matplotlib installed.

## Usage
To run this program, you need to run the neural_network.py file and pass the data files as command line arguments in the correct order. Use this command line:

python neural_network.py 784 30 10 TrainDigitX.csv.gz TrainDigitY.csv.gz TestDigitX.csv.gz TestDigitY.csv.gz PredictDigitY.csv.gz
