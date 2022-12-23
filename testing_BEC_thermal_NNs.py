import tensorflow as tf
import numpy as np
import math
import time
from testing_old_classical_BEC_NNs import Evaluate
from generate_bec_thermal_cloud_nn_data import GenerateBecThermalCloudData
import pickle
import bz2
import os
import pathlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import copy

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

    def __init__(self,x_train,y_train,x_test,y_test):

        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)

        self.validation_x = np.asarray(x_test[len(x_test)//2:])
        self.x_test = np.asarray(x_test[:len(x_test)//2])

        self.validation_y = np.asarray(y_test[len(y_test)//2:])
        self.y_test = np.asarray(y_test[:len(y_test)//2])
        self.compiled_model = self.compile_model()

    def compile_model(self):

        #on 1000 training images, 250 validation images, and 250 test images, model achieved 0.033 root mean sqr error on test images

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(len(self.x_train[0])//2,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(len(self.x_train[0])//4,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(len(self.x_train[0])//8,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(len(self.x_train[0])//16,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(2))

        #0.000001 lr works well for temp only

        optim = tf.keras.optimizers.Adam(learning_rate = 0.001)

        model.compile(optimizer=optim,loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

        return model

class ConvolutionalModel:
    '''

    '''

    #need to put the argparse stuff in this init and in the compilation
    def __init__(self,x_train,y_train,x_test,y_test):

        self.compiled_model = self.compile_model()

        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)

        #should only be getting validation data if that is the validation method of choice
        self.validation_x = np.asarray(x_test[len(x_test)//2:])
        self.x_test = np.asarray(x_test[:len(x_test)//2])

        self.validation_y = np.asarray(y_test[len(y_test)//2:])
        self.y_test = np.asarray(y_test[:len(y_test)//2])

    #more arguments in here
    def compile_model(self):

        model = tf.keras.Sequential()

        #check this line
        model.add(tf.keras.layers.Conv2D(input_shape =(100,100,1) ,kernel_size=3,filters=12,use_bias=False,padding='same'))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.Conv2D(kernel_size=6,filters=24,use_bias=False,padding='same',strides=2))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.Conv2D(kernel_size=6,filters=32,use_bias=False,padding='same',strides=2))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(200,use_bias=False))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.Dense(2))

        optim = tf.keras.optimizers.Adam(learning_rate = 0.001)

        model.compile(optimizer=optim,loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

        return model


class Train:

    def __init__(self,model):
        self.model = model
        self.compiled_model = model.compiled_model
        self.trained_model = self.fit_model(self.compiled_model,self.model)

    @calculate_runtime
    def fit_model(self,compiled_model,model):

        earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 20,restore_best_weights = True)

        compiled_model.fit(model.x_train,model.y_train,epochs=1000,validation_data = (model.validation_x,model.validation_y),callbacks = [earlystopping])

        return compiled_model

class Plot:

    '''
    generates a scatterplot of two arrays
    '''

    def __init__(self,x_list,y_list,x_label,y_label,x_list2 = None ,y_list2 = None,num_plot = 0):
        self.x_list = x_list
        self.y_list = y_list

        self.x_label = x_label
        self.y_label = y_label

        self.x_list2 = x_list2
        self.y_list2 = y_list2

        self.num_plot = num_plot

    def scatter_plot(self):
        plt.scatter(self.x_list,self.y_list)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        if self.x_list2 and self.y_list2:
            plt.scatter(self.x_list2,self.y_list2)

        cwd = pathlib.Path(__file__).parent.resolve()
        plt.savefig(str(cwd)+f"/plots/plot{self.num_plot}.png")
        # plt.show()


def main():

    # num_atoms = 100000
    #
    # trans_temp = (num_atoms/(2*1*1.645))**0.5


    cwd = pathlib.Path(__file__).parent.resolve()
    print(cwd)
    in_file = bz2.BZ2File(str(cwd)+"/generated_data/full_generated_data.bz2",'rb')
    data = pickle.load(in_file)
    in_file.close()


    # data = GenerateBecThermalCloudData(10,5,0,100,100000,trans_temp)

    #use this line if not generating in parallel
    #compiled_model = Model(data.x_train,data.y_train,data.x_test,data.y_test)

    #use if generating data in parallel

    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]

    x_train,x_test = tf.keras.utils.normalize(x_train,axis=1),tf.keras.utils.normalize(x_test,axis=1)

    # print(x_train[0][len(x_train[0])//2,:])

    #make copy of y_test before preprocessed
    tmp_y_test = copy.copy(y_test)

    # preprocess
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.fit_transform(y_test)

    print(np.shape(x_train))


    compiled_model = ConvolutionalModel(x_train,y_train,x_test,y_test)

    trained_model = Train(compiled_model)

    trained_model = trained_model.trained_model

    evaluate = Evaluate(compiled_model,trained_model)

    trained_model.save("BEC_model")


    trained_model = tf.keras.models.load_model('BEC_model')

    predictions = trained_model.predict(x_test)
    temp_predictions = []
    BEC_atoms_predictions = []

    temp_test = []
    BEC_atoms_test = []
    for i in range(len(predictions)):

        temp_predictions.append(predictions[i][0])
        BEC_atoms_predictions.append(predictions[i][1])

        temp_test.append(tmp_y_test[i][0])
        BEC_atoms_test.append(tmp_y_test[i][1])

    # plot = Plot(range(len(temp_test)),temp_test,"obs","temp",range(len(temp_predictions)),temp_predictions)
    # plot.scatter_plot()

    predictions = np.asarray([(temp_predictions[i],BEC_atoms_predictions[i]) for i in range(len(temp_test))])
    predictions = scaler.inverse_transform(predictions)

    temp_predictions = predictions[:,0]
    BEC_atoms_predictions = predictions[:,1]

    plot1 = Plot(range(len(temp_predictions)),temp_predictions,"obs","temp",range(len(temp_test)),temp_test,0)
    plot2 = Plot(range(len(BEC_atoms_predictions)),BEC_atoms_predictions,"obs","num_atoms",range(len(BEC_atoms_test)),BEC_atoms_test,1)

    plot1.scatter_plot()
    plot2.scatter_plot()


if __name__ == "__main__":

    main()
