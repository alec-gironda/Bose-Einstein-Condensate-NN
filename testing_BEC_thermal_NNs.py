import os
#only print error messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import math
import time
import pickle
import bz2
import pathlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
import copy
from sklearn.metrics import mean_squared_error,mean_absolute_error
import argparse

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

        self.x_train = tf.convert_to_tensor(x_train)
        self.y_train = tf.convert_to_tensor(y_train)

        self.validation_x = tf.convert_to_tensor(x_test[len(x_test)//2:])
        self.x_test = tf.convert_to_tensor(x_test[:len(x_test)//2])

        self.validation_y = tf.convert_to_tensor(y_test[len(y_test)//2:])
        self.y_test = tf.convert_to_tensor(y_test[:len(y_test)//2])
        self.compiled_model = self.compile_model()

    def compile_model(self):

        #on 1000 training images, 250 validation images, and 250 test images, model achieved 0.033 root mean sqr error on test images


        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense((len(self.x_train[0])**2)//2,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense((len(self.x_train[0])**2)//4,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense((len(self.x_train[0])**2)//8,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense((len(self.x_train[0])**2)//16,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(2))

        #0.000001 lr works well for temp only

        optim = tf.keras.optimizers.Adam(learning_rate = 0.001)

        model.compile(optimizer=optim,loss='mean_squared_error',metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

        return model

class ConvolutionalModel:

    #need to put the argparse stuff in this init and in the compilation
    def __init__(self,x_train,y_train,x_test,y_test):

        self.x_train = tf.convert_to_tensor(x_train)
        self.y_train = tf.convert_to_tensor(y_train)

        self.validation_x = tf.convert_to_tensor(x_test[len(x_test)//2:])
        self.x_test = tf.convert_to_tensor(x_test[:len(x_test)//2])

        self.validation_y = tf.convert_to_tensor(y_test[len(y_test)//2:])
        self.y_test = tf.convert_to_tensor(y_test[:len(y_test)//2])
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

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense((len(self.x_train[0])**2)//2,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense((len(self.x_train[0])**2)//4,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense((len(self.x_train[0])**2)//8,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense((len(self.x_train[0])**2)//16,activation = tf.nn.relu))
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
        val_loss, val_rmse, val_mae = trained_model.evaluate(compiled_model.x_test,
                                                   compiled_model.y_test)
        return [val_loss, val_rmse, val_mae]

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

    parser = argparse.ArgumentParser(description="arguments for neural network model")

    parser.add_argument(
    "-c",
    "--convolutional",
    action=argparse.BooleanOptionalAction,
    help = "train on convolutional model or load convolutional model"
    )

    parser.add_argument(
    "-l",
    "--load",
    action=argparse.BooleanOptionalAction,
    help = "load model instead of training"
    )

    args = parser.parse_args()

    #load fully generated data

    print("loading data...")
    cwd = pathlib.Path(__file__).parent.resolve()
    in_file = bz2.BZ2File(str(cwd)+"/generated_data/full_generated_data.bz2",'rb')
    data = pickle.load(in_file)
    in_file.close()

    print("data loaded.")

    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]

    '''

    all done on 8000 training images

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

    loss: 1.0181e-05 - mean_squared_error: 1.0181e-05 - mean_absolute_error: 0.0017

    loss: 7.9010e-06 - mean_squared_error: 7.9010e-06 - mean_absolute_error: 0.0018

    '''

    #scale each row in images to be between 0 and 1

    print("preprocessing data...")

    x_scaler = MinMaxScaler(feature_range = (0,1))

    x_train = [x_scaler.fit_transform(x_train[i]) for i in range(len(x_train))]
    x_test = [x_scaler.fit_transform(x_test[i]) for i in range(len(x_test))]

    #make copy of y_test before preprocessing
    tmp_y_test = copy.copy(y_test)

    #scale labels to be between 0 and 1
    y_scaler = MinMaxScaler(feature_range = (0,1))
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.fit_transform(y_test)

    print("data preprocessed.")

    if not args.load:

        print("training model...")

        compiled_model = None

        if args.convolutional:

            compiled_model = ConvolutionalModel(x_train,y_train,x_test,y_test)

        else:

            compiled_model = Model(x_train,y_train,x_test,y_test)

        trained_model = Train(compiled_model)

        trained_model = trained_model.trained_model

        evaluate = Evaluate(compiled_model,trained_model)

        if args.convolutional:

            trained_model.save("BEC_model_conv")

        else:

            trained_model.save("BEC_model")

        print("model trained.")

    trained_model = None

    print("loading trained model...")

    if args.convolutional:

        trained_model = tf.keras.models.load_model('BEC_model_conv')

    else:

        trained_model = tf.keras.models.load_model('BEC_model')

    print("model loaded.")

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

    predictions = np.asarray([(temp_predictions[i],BEC_atoms_predictions[i]) for i in range(len(temp_test))])
    predictions = y_scaler.inverse_transform(predictions)

    temp_predictions = predictions[:,0]
    BEC_atoms_predictions = predictions[:,1]

    #make sure no predictions are less than 0

    temp_predictions = [temp_predictions[i] if temp_predictions[i] > 0 else 0 for i in range(len(temp_predictions))]
    BEC_atoms_predictions = [BEC_atoms_predictions[i] if BEC_atoms_predictions[i] > 0 else 0 for i in range(len(BEC_atoms_predictions))]

    plot1 = Plot(range(len(temp_predictions)),temp_predictions,"obs","temp",range(len(temp_test)),temp_test,0)
    plot2 = Plot(range(len(BEC_atoms_predictions)),BEC_atoms_predictions,"obs","num_atoms",range(len(BEC_atoms_test)),BEC_atoms_test,1)

    plot1.scatter_plot()
    plot2.scatter_plot()

    temp_mse = mean_squared_error(temp_test,temp_predictions)
    atoms_mse = mean_squared_error(BEC_atoms_test,BEC_atoms_predictions)

    print(f"temp mse: {temp_mse}")
    print(f"atoms mse: {atoms_mse}")

    temp_rmse = mean_squared_error(temp_test,temp_predictions,squared = False)
    atoms_rmse = mean_squared_error(BEC_atoms_test,BEC_atoms_predictions,squared = False)

    print(f"temp rmse: {temp_rmse}")
    print(f"atoms rmse: {atoms_rmse}")

    temp_mae = mean_absolute_error(temp_test,temp_predictions)
    atoms_mae = mean_absolute_error(BEC_atoms_test,BEC_atoms_predictions)

    print(f"temp mae: {temp_mae}")
    print(f"atoms mae: {atoms_mae}")

    #check gaussian against this

    

if __name__ == "__main__":

    main()
