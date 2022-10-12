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

    #learning rate

    for i in range(1,6):

        #create a test with decay and without

        learning_rate = str(i*0.01)
        f.write(str(curr_nn_id)+",300,3,60000,"+learning_rate+",1,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,1\n")
        curr_nn_id+=1
        f.write(str(curr_nn_id)+",300,3,60000,"+learning_rate+",0,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,1\n")
        curr_nn_id+=1

    #dropout rates

    for i in range(0,12):

        dropout_rate = str(i*0.05)
        if dropout_rate == '0':
            f.write(str(curr_nn_id)+",300,3,60000,0.01,1,0,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,1\n")

        else:
            f.write(str(curr_nn_id)+",300,3,60000,0.01,1,1,"+dropout_rate+",10,100,sparse_categorical_crossentropy,accuracy,relu,1\n")
        curr_nn_id+=1

    #epochs

    for i in range(1,20):

        epochs = str(i)
        f.write(str(curr_nn_id)+",300,3,60000,0.01,1,1,0.3,"+epochs+",100,sparse_categorical_crossentropy,accuracy,relu,1\n")
        curr_nn_id+=1

    #batch size

    for i in range(10,601,10):

        batch_size = str(i)
        f.write(str(curr_nn_id)+",300,3,60000,0.01,1,1,0.3,10,"+batch_size+",sparse_categorical_crossentropy,accuracy,relu,1\n")
        curr_nn_id+=1

    #non-conv tests

    #hidden units

    for i in range(1,39):

        hidden_units = str(i*20)
        f.write(str(curr_nn_id)+","+hidden_units+",1,60000,0.01,1,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,0\n")
        curr_nn_id+=1

    #layers

    for i in range(1,4):

        layers = str(i)
        f.write(str(curr_nn_id)+",300,"+layers+",60000,0.01,1,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,0\n")
        curr_nn_id+=1

    #training size:

    for i in range(1,13):

        training_size = str(i*5000)
        f.write(str(curr_nn_id)+",300,3,"+training_size+",0.01,1,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,0\n")
        curr_nn_id+=1

    #learning rate

    for i in range(1,6):

        #create a test with decay and without

        learning_rate = str(i*0.01)
        f.write(str(curr_nn_id)+",300,3,60000,"+learning_rate+",1,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,0\n")
        curr_nn_id+=1
        f.write(str(curr_nn_id)+",300,3,60000,"+learning_rate+",0,1,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,0\n")
        curr_nn_id+=1

    #dropout rates

    for i in range(0,12):

        dropout_rate = str(i*0.05)
        if dropout_rate == '0':
            f.write(str(curr_nn_id)+",300,3,60000,0.01,1,0,0.3,10,100,sparse_categorical_crossentropy,accuracy,relu,0\n")

        else:
            f.write(str(curr_nn_id)+",300,3,60000,0.01,1,1,"+dropout_rate+",10,100,sparse_categorical_crossentropy,accuracy,relu,0\n")
        curr_nn_id+=1

    #epochs

    for i in range(1,20):

        epochs = str(i)
        f.write(str(curr_nn_id)+",300,3,60000,0.01,1,1,0.3,"+epochs+",100,sparse_categorical_crossentropy,accuracy,relu,0\n")
        curr_nn_id+=1

    #batch size

    for i in range(10,601,10):

        batch_size = str(i)
        f.write(str(curr_nn_id)+",300,3,60000,0.01,1,1,0.3,10,"+batch_size+",sparse_categorical_crossentropy,accuracy,relu,0\n")
        curr_nn_id+=1
