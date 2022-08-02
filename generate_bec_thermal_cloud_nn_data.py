from generate_classical_tmp_nn_data import GenerateData
import numpy as np
import math
import random
import time

class GenerateBecThermalCloudData(GenerateData):

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

    @calculate_runtime
    def __init__(self,training_size,test_size,noise_spread,resolution_length,num_atoms,trans_temp):
        super().__init__(training_size,test_size,noise_spread,resolution_length)
        self.num_atoms = num_atoms
        self.trans_temp = trans_temp
        self.training_data = self.generate_data(training_size,resolution_length,noise_spread,num_atoms,trans_temp)
        self.test_data = self.generate_data(test_size,resolution_length,noise_spread,num_atoms,trans_temp)

        self.x_train = self.training_data[0]
        self.y_train = self.training_data[1]

        self.x_test = self.test_data[0]
        self.y_test = self.test_data[1]

        self.data_tup = ((self.x_train,self.y_train),(self.x_test,self.y_test))


    def generate_noise_image(self,temp,length,noise_spread,num_atoms,trans_temp):
        n_arr = np.zeros((length,length))
        n_arr = n_arr.tolist()
        for x in range(length):
            for y in range(length):
                temp_ratio = temp/trans_temp
                if temp_ratio > 1:
                    temp_ratio =1
                n_arr[x][y] = (num_atoms*(1-(temp_ratio)**2)*(1/math.pi)*math.e**(-(x-length//2)**2)*math.e**(-(y-length//2)**2)
                              +num_atoms*(temp_ratio)**2*(1/(2*math.pi*temp))
                              *math.e**((-(x-length//2)**2)/(2*temp))*math.e**((-(y-length//2)**2)/(2*temp))+np.random.normal(0,noise_spread))
        return n_arr
    def generate_data(self,size,length,noise_spread,num_atoms,trans_temp):

        x_data = []
        y_data = []

        for i in range(size//4):
            curr_img = self.generate_noise_image(87,length,noise_spread,num_atoms,trans_temp)
            x_data.append(curr_img)
            y_data.append(0)
        for i in range(size//4):
            curr_img = self.generate_noise_image(125,length,noise_spread,num_atoms,trans_temp)
            x_data.append(curr_img)
            y_data.append(1)
        for i in range(size//4):
            curr_img = self.generate_noise_image(150,length,noise_spread,num_atoms,trans_temp)
            x_data.append(curr_img)
            y_data.append(2)
        for i in range(size//4):
            curr_img = self.generate_noise_image(280,length,noise_spread,num_atoms,trans_temp)
            x_data.append(curr_img)
            y_data.append(3)

        x_data,y_data = self.shuffle_data(x_data,y_data)

        return x_data,y_data
