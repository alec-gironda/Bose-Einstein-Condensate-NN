import tensorflow as tf
import numpy as np
import math
import time
from testing_old_classical_BEC_NNs import Evaluate
from generate_bec_thermal_cloud_nn_data import GenerateBecThermalCloudData

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

        self.compiled_model = self.compile_model()

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

    def compile_model(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(5000))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dense(2000))
        model.add(tf.keras.layers.BatchNormalization(center=True,scale=False))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dense(1))

        # optim = tf.keras.optimizers.Adam(learning_rate=0.1)
        model.compile(optimizer="adam",loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])

        return model

class Train:

    def __init__(self,model):
        self.model = model
        self.compiled_model = model.compiled_model
        self.trained_model = self.fit_model(self.compiled_model,self.model)

    @calculate_runtime
    def fit_model(self,compiled_model,model):

        compiled_model.fit(model.x_train,model.y_train,epochs=10)

        return compiled_model


def main():

    num_atoms = 100000

    trans_temp = (num_atoms/(2*1*1.645))**0.5

    data = GenerateBecThermalCloudData(100,50,0,100,100000,trans_temp)

    compiled_model = Model(data.x_train,data.y_train,data.x_test,data.y_test)

    trained_model = Train(compiled_model)

    trained_model = trained_model.trained_model


    evaluate = Evaluate(compiled_model,trained_model)


if __name__ == "__main__":

    main()
