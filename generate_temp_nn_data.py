import numpy as np
import math
import random

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

    def __init__(self,training_size,test_size):

        x_train,y_train = self.generate_data(training_size)
        x_test,y_test = self.generate_data(test_size)

    def generate_noise_image(self,temp,length):
        n_arr = np.zeros((length,length))
        n_arr = n_arr.tolist()
        for x in range(length):
            for y in range(length):
                n_arr[x][y] = (1/temp)*math.e**(-(x-length//2)**2/temp)*math.e**(-(y-length//2)**2/temp)+np.random.normal(0,0.01)
        return n_arr

    @calculate_runtime
    def generate_data(self,size):

        x_data = []
        y_data = []

        for i in range(size//2):
            curr_img = self.generate_noise_image(0.5,29)
            x_data.append(curr_img)
            y_data.append(0.5)
        for i in range(size//2):
            curr_img = self.generate_noise_image(2,29)
            x_data.append(curr_img)
            y_data.append(2)

        self.shuffle_data(x_data,y_data)

        return x_data,y_data

    def shuffle_data(self,x_data,y_data):
        shuffle_list = list(zip(x_data,y_data))
        random.shuffle(shuffle_list)
        x_data, y_data = zip(*shuffle_list)
        x_data, y_data = list(x_data), list(y_data)

if __name__ == "__main__":
    generate = GenerateData(10,5)
    print('hi')
