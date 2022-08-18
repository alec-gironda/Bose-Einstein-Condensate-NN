from generate_classical_tmp_nn_data import GenerateData
from generate_image_with_sampling import GenerateSampledImage
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

        sample = GenerateSampledImage(temp,length,num_atoms)
        return sample.generated_image

    def generate_data(self,size,length,noise_spread,num_atoms,trans_temp):

        x_data = []
        y_data = []

        #generating discrete temperatures

        # for i in range(size//4):
        #     curr_img = self.generate_noise_image(87,length,noise_spread,num_atoms,trans_temp)
        #     x_data.append(curr_img)
        #     y_data.append(0)
        # for i in range(size//4):
        #     curr_img = self.generate_noise_image(125,length,noise_spread,num_atoms,trans_temp)
        #     x_data.append(curr_img)
        #     y_data.append(1)
        # for i in range(size//4):
        #     curr_img = self.generate_noise_image(170,length,noise_spread,num_atoms,trans_temp)
        #     x_data.append(curr_img)
        #     y_data.append(2)
        # for i in range(size//4):
        #     curr_img = self.generate_noise_image(280,length,noise_spread,num_atoms,trans_temp)
        #     x_data.append(curr_img)
        #     y_data.append(3)

        #generating continuous temperatures

        for i in range(size):
            #temp = np.random.uniform(87,348)
            temp = np.random.uniform(87,348)
            num_BEC_atoms = 0
            if temp < trans_temp:
                num_BEC_atoms = int(num_atoms*(1-(temp/trans_temp)**2))
            curr_img = self.generate_noise_image(temp,length,noise_spread,num_atoms,trans_temp)
            x_data.append(curr_img)
            y_data.append((temp,num_BEC_atoms))

        x_data,y_data = self.shuffle_data(x_data,y_data)

        return x_data,y_data
