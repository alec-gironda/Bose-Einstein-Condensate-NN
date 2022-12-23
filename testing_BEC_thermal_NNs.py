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
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
import copy
from sklearn.metrics import mean_squared_error,mean_absolute_error


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

        model.compile(optimizer=optim,loss='mean_squared_error',metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

        return model

class ConvolutionalModel:

    #need to put the argparse stuff in this init and in the compilation
    def __init__(self,x_train,y_train,x_test,y_test):

        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)

        #should only be getting validation data if that is the validation method of choice
        self.validation_x = np.asarray(x_test[len(x_test)//2:])
        self.x_test = np.asarray(x_test[:len(x_test)//2])

        self.validation_y = np.asarray(y_test[len(y_test)//2:])
        self.y_test = np.asarray(y_test[:len(y_test)//2])

        self.compiled_model = self.compile_model()

    #more arguments in here
    def compile_model(self):

        model = tf.keras.Sequential()

        #check this line
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(100, 100, 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(len(self.x_train[0])//2,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(len(self.x_train[0])//4,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(len(self.x_train[0])//8,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(len(self.x_train[0])//16,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(2))

        model.summary()

        optim = tf.keras.optimizers.Adam(learning_rate = 0.001)

        model.compile(optimizer=optim,loss='mean_squared_error',metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

        return model


class Train:

    def __init__(self,model):
        self.model = model
        self.compiled_model = model.compiled_model
        self.trained_model = self.fit_model(self.compiled_model,self.model)

    @calculate_runtime
    def fit_model(self,compiled_model,model):

        earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 50,restore_best_weights = True)

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

    '''

    x data scaled with tensorflow normalization

    y scaled with standardscaler

     l 0.0014 rmse 0.0373 mae 0.0218

    x data with MinMaxScaler

    y scaled with standardscaler

    loss: 7.7384e-04 - root_mean_squared_error: 0.0278 - mean_absolute_error: 0.0189

    x data scaled with MinMaxScaler

    y scaled with MinMaxScaler

    loss: 9.4292e-05 - root_mean_squared_error: 0.0097 - mean_absolute_error: 0.0063

    cnn

    loss: 1.1483e-05 - root_mean_squared_error: 0.0034 - mean_absolute_error: 0.0020


    '''

    # x_scaler = MaxAbsScaler()
    x_scaler = MinMaxScaler(feature_range = (0,1))

    x_train = [x_scaler.fit_transform(x_train[i]) for i in range(len(x_train))]
    x_test = [x_scaler.fit_transform(x_test[i]) for i in range(len(x_test))]
    # x_train,x_test = tf.keras.utils.normalize(x_train,axis=1),tf.keras.utils.normalize(x_test,axis=1)

    # print(x_train[0][len(x_train[0])//2,:])

    #make copy of y_test before preprocessed
    tmp_y_test = copy.copy(y_test)

    # preprocess
    y_scaler = MinMaxScaler(feature_range = (0,1))
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.fit_transform(y_test)

    print(np.shape(x_test))



    compiled_model = ConvolutionalModel(x_train,y_train,x_test,y_test)

    trained_model = Train(compiled_model)

    trained_model = trained_model.trained_model

    evaluate = Evaluate(compiled_model,trained_model)

    trained_model.save("BEC_model")



    trained_model = tf.keras.models.load_model('BEC_model')

    x_test = np.asarray(x_test)

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
    predictions = y_scaler.inverse_transform(predictions)

    temp_predictions = predictions[:,0]
    BEC_atoms_predictions = predictions[:,1]

    plot1 = Plot(range(len(temp_predictions)),temp_predictions,"obs","temp",range(len(temp_test)),temp_test,0)
    plot2 = Plot(range(len(BEC_atoms_predictions)),BEC_atoms_predictions,"obs","num_atoms",range(len(BEC_atoms_test)),BEC_atoms_test,1)

    plot1.scatter_plot()
    plot2.scatter_plot()

    temp_mse = mean_squared_error(temp_test,temp_predictions)
    atoms_mse = mean_squared_error(BEC_atoms_test,BEC_atoms_predictions)

    print(f"temp mse: {temp_mse}")
    print(f"atoms mse: {atoms_mse}")

    temp_mae = mean_absolute_error(temp_test,temp_predictions)
    atoms_mae = mean_absolute_error(BEC_atoms_test,BEC_atoms_predictions)

    print(f"temp mae: {temp_mae}")
    print(f"atoms mae: {atoms_mae}")

    for indx,pred in enumerate(BEC_atoms_predictions):
        if abs(int(pred-BEC_atoms_test[indx])) > 300:
            print(int(pred-BEC_atoms_test[indx]),indx,int(pred),int(BEC_atoms_test[indx]),int(temp_test[indx]))


if __name__ == "__main__":

    main()
