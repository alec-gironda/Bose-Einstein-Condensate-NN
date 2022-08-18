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

        self.validation_x = x_test[len(x_test)//2:]
        self.x_test = x_test[:len(x_test)//2]

        self.validation_y = y_test[len(y_test)//2:]
        self.y_test = y_test[:len(y_test)//2]

    def compile_model(self):

        #on 1000 training images, 250 validation images, and 250 test images, model achieved 0.033 root mean sqr error on test images

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(5000,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(2,activation= tf.nn.relu))

        #0.000001 lr works well for temp only

        optim = tf.keras.optimizers.Adam(learning_rate = 0.00001)

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


def main():

    num_atoms = 100000

    trans_temp = (num_atoms/(2*1*1.645))**0.5

    data = GenerateBecThermalCloudData(10,5,0,100,100000,trans_temp)

    compiled_model = Model(data.x_train,data.y_train,data.x_test,data.y_test)

    trained_model = Train(compiled_model)

    trained_model = trained_model.trained_model


    evaluate = Evaluate(compiled_model,trained_model)

    print(evaluate)


if __name__ == "__main__":

    main()
