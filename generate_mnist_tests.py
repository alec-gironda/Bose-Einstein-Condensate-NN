'''

dataset,hidden_units,layers,training_size,
learning_rate,decay_lr,dropout,dropout_size,epochs,
batch_size,loss,metrics,activation,convolutional):

'''

#7,350,1,60000,0.01,1,0,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,1

curr_nn_id = 1
with open('mnist_tests.txt', 'w') as f:

    #conv tests

    #hidden units

    for i in range(1,39):

        hidden_units = str(i*20)
        f.write(str(curr_nn_id)+","+hidden_units+",1,60000,0.01,1,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,1\n")
        curr_nn_id+=1

    #layers

    for i in range(1,4):

        layers = str(i)
        f.write(str(curr_nn_id)+",300,"+layers+",60000,0.01,1,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,1\n")
        curr_nn_id+=1

    #training size:

    for i in range(1,13):

        training_size = str(i*5000)
        f.write(str(curr_nn_id)+",300,3,"+training_size+",0.01,1,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,1\n")
        curr_nn_id+=1
