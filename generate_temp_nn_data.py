import numpy as np
import math
import random
import time

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

class GenerateData:

    @calculate_runtime
    def __init__(self,training_size,test_size,noise_spread,resolution_length):

        self.noise_spread = noise_spread
        self.resolution_length = resolution_length

        self.training_data = self.generate_data(training_size,resolution_length,noise_spread)
        self.x_train = self.training_data[0]
        self.y_train = self.training_data[1]

        self.test_data = self.generate_data(test_size,resolution_length,noise_spread)
        self.x_test = self.test_data[0]
        self.y_test = self.test_data[1]

        self.data_tup = ((self.x_train,self.y_train),(self.x_test,self.y_test))

    def generate_noise_image(self,temp,length,noise_spread):
        n_arr = np.zeros((length,length))
        n_arr = n_arr.tolist()
        for x in range(length):
            for y in range(length):
                n_arr[x][y] = (1/temp)*math.e**(-(x-length//2)**2/temp)*math.e**(-(y-length//2)**2/temp)+np.random.normal(0,noise_spread)
        return n_arr

    def generate_data(self,size,length,noise_spread):

        x_data = []
        y_data = []

        for i in range(size//4):
            curr_img = self.generate_noise_image(0.5,length,noise_spread)
            x_data.append(curr_img)
            y_data.append(0)
        for i in range(size//4):
            curr_img = self.generate_noise_image(1,length,noise_spread)
            x_data.append(curr_img)
            y_data.append(1)
        for i in range(size//4):
            curr_img = self.generate_noise_image(1.5,length,noise_spread)
            x_data.append(curr_img)
            y_data.append(2)
        for i in range(size//4):
            curr_img = self.generate_noise_image(2,length,noise_spread)
            x_data.append(curr_img)
            y_data.append(3)

        x_data,y_data = self.shuffle_data(x_data,y_data)

        return x_data,y_data

    def shuffle_data(self,x_data,y_data):
        shuffle_list = list(zip(x_data,y_data))
        random.shuffle(shuffle_list)
        x_data, y_data = zip(*shuffle_list)
        x_data, y_data = list(x_data), list(y_data)
        return x_data,y_data
