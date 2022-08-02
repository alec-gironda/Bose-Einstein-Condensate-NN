from generate_classical_tmp_nn_data import GenerateData
import numpy as np
import math
import random
import time

class GenerateBecThermalCloudData(GenerateData):

    def __init__(self,num_atoms,trans_temp):
        super().__init__
        self.num_atoms = num_atoms
        self.trans_temp = trans_temp

    def generate_noise_image(self,temp,length,noise_spread,num_atoms,trans_temp):
        n_arr = np.zeros((length,length))
        n_arr = n_arr.tolist()
        for x in range(length):
            for y in range(length):
                n_arr[x][y] = (num_atoms*(1-(temp/trans_temp)**2)*(1/math.pi)*math.e**(-x**2)*math.e**(-y**2)
                              +num_atoms*(temp/trans_temp)**2*((x*y)/(2*math.pi*temp))
                              *math.e**((-x**2)/(2*math.pi*temp))*math.e**((-y**2)/(2*math.pi*temp)))#+np.random.normal(0,noise_spread)
        return n_arr
