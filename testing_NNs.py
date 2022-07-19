import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from NN_database import Database
import pandas as pd


def calculate_runtime(func):
    '''
    decorator to calculate the runtime of functions
    while still returning their output
    '''
    def wrapper(*args,**kwargs):
        start_time = time.time()

        out = func(*args,**kwargs)

        print("--- %s seconds ---" % (time.time() - start_time))

        return out
    return wrapper


class Model:
    '''
    takes information on dataset and neural network parameters.
    compiles a model.

    imported data must be in (x_train,y_train),(x_test,y_test) = data format


    can try different optimizers
    can try different activation functions

    '''

    def __init__(self,dataset,hidden_units,layers,training_size,
                learning_rate,decay_lr,dropout,dropout_size,epochs,
                batch_size,loss,metrics,activation):

        #read in inputs
        self.dataset = dataset
        self.hidden_units = hidden_units
        self.layers = layers
        self.training_size = training_size
        self.learning_rate = learning_rate
        self.decay_lr = decay_lr
        self.dropout = dropout
        self.dropout_size = dropout_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.metrics = [metrics]
        self.activation = activation

        processed_data = self.process_dataset_input(self.dataset)

        self.x_train = tf.keras.utils.normalize(processed_data[0][:self.training_size],axis=1)
        self.y_train = processed_data[1][:self.training_size]

        self.steps_per_epoch = len(self.x_train)//self.batch_size

        self.x_test = tf.keras.utils.normalize(processed_data[2],axis=1)
        self.y_test = processed_data[3]

        #convolutional NN stuff
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28, 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28, 1))

        self.compiled_model = self.compile_model(self.hidden_units,
                                                 self.learning_rate,
                                                 self.dropout,self.dropout_size,
                                                 self.loss,self.metrics,self.activation,
                                                 self.layers)

    def process_dataset_input(self,dataset):
        '''

        processes data with [2,2] shape

        (x_train,y_train),(x_test,y_test) = data

        '''

        x_train = dataset[0][0]
        y_train = dataset[0][1]

        x_test = dataset[1][0]
        y_test = dataset[1][1]

        return [x_train,y_train,x_test,y_test]

    def compile_model(self,hidden_units,learning_rate,
                      dropout,dropout_size,loss,metrics,activation,layers):

        '''
        the actual architecture of the model

        '''


        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Flatten())
        # curr_hidden_units = hidden_units
        # for layer in range(layers):
        #     model.add(tf.keras.layers.Dense(curr_hidden_units,activation=activation))
        #     if dropout:
        #         model.add(tf.keras.layers.Dropout(dropout_size))
        #         curr_hidden_units //=2
        #         if curr_hidden_units <10:
        #             curr_hidden_units = 10
        # model.add(tf.keras.layers.Dense(10,activation='softmax'))
        #
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        #
        # model.compile(optimizer=optimizer,loss=loss,
        #               metrics=metrics)

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(kernel_size=3,filters=12,use_bias=False,padding='same'))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation(activation))

        model.add(tf.keras.layers.Conv2D(kernel_size=6,filters=24,use_bias=False,padding='same',strides=2))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation(activation))

        model.add(tf.keras.layers.Conv2D(kernel_size=6,filters=32,use_bias=False,padding='same',strides=2))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation(activation))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(200,use_bias=False))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation(activation))

        model.add(tf.keras.layers.Dropout(dropout_size))
        model.add(tf.keras.layers.Dense(10,activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

        return model


class Train:
    '''

    trains the model on the training data

    '''

    def __init__(self,model):
        self.model = model
        self.compiled_model = model.compiled_model
        self.decay_lr = None
        if model.decay_lr:
            self.decay_lr = self.lr_decay_function()

        self.trained_model = self.fit_model(self.compiled_model,self.model)

    def lr_decay_function(self):
        '''
        function to decay the learning rate as epoch number increases

        '''
        return [tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 *
                                                        math.pow(0.6,epoch),
                                                        verbose=True)]
    @calculate_runtime
    def fit_model(self,compiled_model,model):
        compiled_model.fit(model.x_train,model.y_train,epochs=model.epochs,steps_per_epoch = model.steps_per_epoch,
                                                callbacks=self.decay_lr)

        return compiled_model

class Evaluate:

    '''
    gets loss and accuracy of the model based on the validation(test) set

    '''

    def __init__(self,compiled_model,trained_model):
        self.compiled_model = compiled_model
        self.trained_model = trained_model

        statistics = self.get_statistics(self.compiled_model,
                                            self.trained_model)
        self.val_loss = statistics[0]
        self.val_acc = statistics[1]

    def get_statistics(self,compiled_model,trained_model):
        val_loss, val_acc = trained_model.evaluate(compiled_model.x_test,
                                                   compiled_model.y_test)
        return [val_loss,val_acc]

class Plot:

    '''
    generates a scatterplot of two arrays

    '''

    def __init__(self,x_list,y_list,x_label,y_label):
        self.x_list = x_list
        self.y_list = y_list

        self.x_label = x_label
        self.y_label = y_label

    def scatter_plot(self):
        plt.scatter(self.x_list,self.y_list)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()

def main():

    #check CPU performance vs GPU
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    db = Database()

    connection = db.create_db_connection("140.233.160.216", "agironda", "phys_research1", "mnist_db")

    mnist = tf.keras.datasets.mnist
    mnist = mnist.load_data()
    mnist_list = [mnist]

    populate_mnist_table = "INSERT INTO mnist VALUES \n"

    with open('mnist_tests.txt') as f:
        lines = f.readlines()
    for line in lines:
        mnist_list.extend(line.rstrip('\n').split(','))
        input = mnist_list
        mnist_list = [mnist]

        '''

        dataset,hidden_units,layers,training_size,
        learning_rate,decay_lr,dropout,dropout_size,epochs,
        batch_size,loss,metrics,activation):

        '''

        compiled_model = Model(input[0],int(input[2]),int(input[3]),int(input[4]),
                               float(input[5]),bool(input[6]),bool(input[7]),float(input[8]),
                               int(input[9]),int(input[10]),input[11],input[12],input[13])

        trained_model = Train(compiled_model)
        trained_model = trained_model.trained_model

        evaluate = Evaluate(compiled_model,trained_model)

        data_tuple = (int(input[1]),int(input[2]),int(input[3]),int(input[4]),
                      float(input[5]),bool(input[6]),bool(input[7]),float(input[8]),
                      int(input[9]),int(input[10]),input[11],input[12],input[13],evaluate.val_acc)

        populate_mnist_table += str(data_tuple) + ",\n"

    populate_mnist_table = populate_mnist_table[:-2] + ";"

    db.execute_query(connection,populate_mnist_table)

    # training_size_plot = Plot(layer_list,val_acc_list,"number of layers","accuracy")
    # training_size_plot.scatter_plot()

    #/usr/local/mysql/bin/mysql -u root -p


if __name__ == "__main__":

    main()
