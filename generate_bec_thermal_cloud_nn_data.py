from generate_classical_tmp_nn_data import GenerateData
from generate_image_with_sampling import GenerateSampledImage
import numpy as np
import math
import random
import time

class GenerateBecThermalCloudData(GenerateData):
    '''
        attributes:

            num_atoms (int) : total number of atoms to be generated in the images
            trans_temp (float) : transition temperature
            dimensions (int) : dimension of the data (2D or 3D)
            training_data (tuple(list[list[list[int]]],list[tuple(float,int)])) : tuple including all data to train model
            test_data (tuple(list[list[list[int]]],list[tuple(float,int)])) : tuple in
            x_train (list[list[list[int]]]) :
            y_train (list[tuple(float,int)]) :
            x_test (list[list[list[int]]]) :
            y_test (list[tuple(float,int)]) :
            data_tup (tuple(tuple(x_train,y_train),tuple(x_test,y_test))) :

        methods:

            generate_noise_image: generates a single image using probability sampling
            generate_data: draws random temperatures from a uniform distribution to
                           generate a list of images and labels including temperature
                           and number of atoms in the BEC
    '''

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
    def __init__(self,training_size,test_size,noise_spread,resolution_length,num_atoms,trans_temp,dimensions,seed):
        super().__init__(training_size,test_size,noise_spread,resolution_length)
        self.num_atoms = num_atoms
        self.trans_temp = trans_temp
        self.dimensions = dimensions
        self.seed = seed
        self.training_data = self.generate_data(training_size,resolution_length,noise_spread,num_atoms,trans_temp,dimensions,seed)
        self.test_data = self.generate_data(test_size,resolution_length,noise_spread,num_atoms,trans_temp,dimensions,seed)

        self.x_train = self.training_data[0]
        self.y_train = self.training_data[1]

        self.x_test = self.test_data[0]
        self.y_test = self.test_data[1]

        self.data_tup = ((self.x_train,self.y_train),(self.x_test,self.y_test))


    def generate_noise_image(self,temp,length,noise_spread,num_atoms,trans_temp,dimensions):

        sample = GenerateSampledImage(temp,length,num_atoms,dimensions)
        return sample.generated_image

    def generate_data(self,size,length,noise_spread,num_atoms,trans_temp,dimensions,seed):

        np.random.seed(seed)

        x_data = []
        y_data = []

        for i in range(size):
            temp = np.random.uniform(trans_temp//2,int(trans_temp*2))
            num_BEC_atoms = 0
            if temp < trans_temp:
                num_BEC_atoms = int(num_atoms*(1-(temp/trans_temp)**dimensions))
            curr_img = self.generate_noise_image(temp,length,noise_spread,num_atoms,trans_temp,dimensions)
            x_data.append(curr_img)
            y_data.append((temp,num_BEC_atoms))

        x_data,y_data = self.shuffle_data(x_data,y_data)

        return x_data,y_data
