import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

#imported data must be in (x_train,y_train),(x_test,y_test) = data format

class Model:
    '''
    takes information on dataset and neural network parameters.
    generates a model.


    can try different optimizers
    can try different activation functions

    '''

    def __init__(self,dataset,hidden_units,layers,training_size,
                learning_rate,decay_lr,dropout,dropout_size,epochs,
                batch_size,loss,metrics):

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
        self.metrics = metrics

        processed_data = self.process_dataset_input(self.dataset)

        self.x_train = processed_data[0][:self.training_size+1]
        self.y_train = processed_data[1][:self.training_size+1]

        self.steps_per_epoch = len(self.x_train)//self.batch_size

        self.x_test = processed_data[2]
        self.y_test = processed_data[3]

        self.compiled_model = self.compile_model(self.hidden_units,
                                                 self.learning_rate,
                                                 self.dropout,self.dropout_size,
                                                 self.loss,self.metrics)

    def process_dataset_input(self,dataset):

        x_train = dataset[0][0]
        y_train = dataset[0][1]

        x_test = dataset[1][0]
        y_test = dataset[1][1]

        return [x_train,y_train,x_test,y_test]

    def compile_model(self,hidden_units,learning_rate,
                      dropout,dropout_size,loss,metrics):
        '''
        add layers, but start with one for now
        '''

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(hidden_units,activation=tf.nn.relu))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout_size))
        model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,loss=loss,
                      metrics=metrics)

        return model

class Train:

    def __init__(self,model):
        self.model = model
        self.compiled_model = model.compiled_model
        self.decay_lr = None
        if model.decay_lr:
            self.decay_lr = self.lr_decay_function()

        self.trained_model = self.fit_model(self.compiled_model,self.model)

    def lr_decay_function(self):
        return [tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 *
                                                        math.pow(0.6,epoch),
                                                        verbose=True)]
    def fit_model(self,compiled_model,model):
        compiled_model.fit(model.x_train,model.y_train,epochs=model.epochs,
                  steps_per_epoch = model.steps_per_epoch,
                  callbacks=self.decay_lr)
        return compiled_model

class Evaluate:

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



if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    mnist = mnist.load_data()

    #(self,dataset,hidden_units,layers,training_size,
    #learning_rate,decay_lr,dropout,dropout_size,epochs,
    #batch_size,loss,metrics):

    compiled_model = Model(mnist,200,1,60000,0.01,False,False,0,10,100,
                 'sparse_categorical_crossentropy',['accuracy'])

    trained_model = Train(compiled_model)
    trained_model = trained_model.trained_model

    evaluate = Evaluate(compiled_model,trained_model)

    print("loss: ",evaluate.val_loss,'\n',"acc: ",evaluate.val_acc)
