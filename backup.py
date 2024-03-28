# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:28:28 2024

@author: dan
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:22:57 2024

@author: dan
"""
import random
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import math
from scipy import stats
from scipy.stats import norm


class MultiLayerPerceptron(BaseEstimator, ClassifierMixin): 
    def __init__(self, params=None):     
        if (params == None):
            self.inputLayer = 10                       # Input Layer
            self.hiddenLayer = 50                      # Hidden Layer
            self.OutputLayer = 1                       # Outpuy Layer
            self.learningRate = 0.001                  # Learning rate
            self.max_epochs = 50                      # Epochs
            self.BiasHiddenValue = -1                   # Bias HiddenLayer initial
            self.BiasOutputValue = -1                  # Bias OutputLayer initial
            self.activ = self.activation['linear'] # Activation function
            self.deriv = self.derivative['linear']
        else:
            self.inputLayer = params['InputLayer']
            self.hiddenLayer = params['HiddenLayer']
            self.OutputLayer = params['OutputLayer']
            self.learningRate = params['LearningRate']
            self.max_epochs = params['Epocas']
            self.BiasHiddenValue = params['BiasHiddenValue']
            self.BiasOutputValue = params['BiasOutputValue']
            self.activ = self.activation[params['ActivationFunction']]
            self.deriv = self.derivative[params['ActivationFunction']]
        
        'Starting Bias and Weights'
        self.WEIGHT_hidden = self.starting_weights(self.hiddenLayer, self.inputLayer)
        self.WEIGHT_output = self.starting_weights(self.OutputLayer, self.hiddenLayer)
        self.BIAS_hidden = np.array([self.BiasHiddenValue for i in range(self.hiddenLayer)])
        self.BIAS_output = np.array([self.BiasOutputValue for i in range(self.OutputLayer)])
        self.output_number = 1 #defines how many numbers are in the output
        
        
    pass
    
    def starting_weights(self, x, y):
        return [[2  * random.random() - 1 for i in range(x)] for j in range(y)]

    activation = {
         'sigmoid': (lambda x: 1/(1 + np.exp(-x))),
            'tanh': (lambda x: np.tanh(x)),
            'Relu': (lambda x: x*(x > 0)),
            'linear': (lambda x: x)
               }
    derivative = {
         'sigmoid': (lambda x: x*(1-x)),
            'tanh': (lambda x: 1-x**2),
            'Relu': (lambda x: 1 * (x>0)),
            'linear': (lambda x: np.ones_like(x))
               }
 
    def Backpropagation_Algorithm(self, inputs, ERROR_output):

        'Stage 1 - Error: OutputLayer'
        DELTA_output = ((-1)*(ERROR_output) * self.deriv(self.OUTPUT_L2))

        'Stage 2 - Update weights OutputLayer and HiddenLayer'
        for i in range(self.hiddenLayer):
            for j in range(self.OutputLayer):
                self.WEIGHT_output[i][j] -= (self.learningRate * (DELTA_output[j] * self.OUTPUT_L1[i]))
                self.BIAS_output[j] -= (self.learningRate * DELTA_output[j])
          
        'Stage 3 - Error: HiddenLayer'
        delta_hidden = np.matmul(self.WEIGHT_output, DELTA_output)* self.deriv(self.OUTPUT_L1)

        'Stage 4 - Update weights HiddenLayer and InputLayer(x)'
        for i in range(self.OutputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= (self.learningRate * (delta_hidden[j] * inputs[i]))
                self.BIAS_hidden[j] -= (self.learningRate * delta_hidden[j])
                
    def show_err_graphic(self,error,epochs):
        plt.figure(figsize=(9,4))
        plt.plot(epochs, error, "m-",color="b", marker=11)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Absolute Error ");
        plt.title("Error Minimization")
        plt.show()

    def predict(self, X, y):
        'Returns the predictions for every element of X'
        my_predictions = []
        'Forward Propagation'
        forward = np.matmul(X,self.WEIGHT_hidden) + self.BIAS_hidden
        forward = np.matmul(forward, self.WEIGHT_output) + self.BIAS_output
                                 
        for i in forward:
            my_predictions.append(i)
            
        array_score = []
        for i in range(len(my_predictions)):
            error = y[i]-my_predictions[i]
            #Check if the prediction is close to the actual
            if np.abs(error) <0.05:
                correct = 1
            else:
                correct=0
            array_score.append(correct)
            
        total_accuracy = (sum(array_score) / len(array_score))*100
        print("Total Accuracy:", total_accuracy)

        return my_predictions, total_accuracy

    def fit(self, X, y):  
        count_epoch = 1

        epoch_array = []
        error_array = []
        W0 = []
        W1 = []
        while(count_epoch <= self.max_epochs):
            for inputs, target in zip(X,y): 
                'Stage 1 - (Forward Propagation)'
                self.OUTPUT_L1 = self.activ((np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden.T))
                self.OUTPUT_L2 = self.activ((np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output.T))

                #Find the difference between the correct datapoint and the NN output
                total_error = target - self.OUTPUT_L2

                'Backpropagation : Update Weights'
                self.Backpropagation_Algorithm(inputs, total_error)
                
            if((count_epoch % 50 == 0)or(count_epoch == 1)):
                print("Epoch ", count_epoch, "- Total Error: ",total_error)

                error_array.append(total_error)
                epoch_array.append(count_epoch)
                
            W0.append(self.WEIGHT_hidden)
            W1.append(self.WEIGHT_output)
             
            count_epoch += 1
            
        self.show_err_graphic(error_array,epoch_array)
        
        plt.plot(W0[0])
        plt.title('Weight Hidden update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3', 'neuron4', 'neuron5'])
        plt.ylabel('Value Weight')
        plt.show()
        
        plt.plot(W1[0])
        plt.title('Weight Output update during training')
        plt.legend(['neuron1'])
        plt.ylabel('Value Weight')
        plt.show()

        return self

def gaussian(x, mu, sig):
    return (
        1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    )
      
def randomly_generate_data_gaussian(num_samples):
    y_train = []
    x_train = []
    mean =  0
    x_values = np.linspace(-3, 3, 10)
    
    for i in range(num_samples):
        #standard_deviation = random.uniform(0.5, 1.5) #Produces a random float value
        standard_deviation = random.randrange(1,5) #Produces a integer between the range
        #gaussian_vector = gaussian(x_values, mean, standard_deviation)
        rv = norm(loc = mean, scale = standard_deviation)
        gaussian_vector = rv.pdf(x_values)
        y_train.append([standard_deviation])  # Append standard_deviation as a new row
        x_train.append(list(gaussian_vector))  # Append gaussian vector as a new row
    #plt.plot(x_values,gaussian_vector)
    return np.array(x_train), np.array(y_train)
    

def randomly_generate_data_uniform(num_samples):
    y_train = []
    x_train = []
    mean =  0
    x_values = np.linspace(0, 1, 10)
    
    for i in range(num_samples):
        #standard_deviation = random.uniform(0.5, 1.5) #Produces a random float value
        multi_num = random.randrange(1,8) #Produces a integer between the range
        numbers = multi_num*x_values

        y_train.append([multi_num])  # Append standard_deviation as a new row
        x_train.append(list(numbers))  # Append gaussian vector as a new row
    #plt.plot(x_values,gaussian_vector)
    return np.array(x_train), np.array(y_train)
#%%
#dictionary = {'InputLayer':4, 'HiddenLayer':5, 'OutputLayer':1,
#              'Epochs':200, 'LearningRate':0.005,'BiasHiddenValue':-1, 
#              'BiasOutputValue':-1, 'ActivationFunction':'sigmoid'}
#Set seed for repeated trials
random.seed(471)  
x_train, y_train = randomly_generate_data_uniform(1000)
x_test, y_test = randomly_generate_data_uniform(100)
#%%
Perceptron = MultiLayerPerceptron()
Perceptron.fit(x_train,y_train)
#%%
predictions, accuracy = Perceptron.predict(x_test, y_test)
