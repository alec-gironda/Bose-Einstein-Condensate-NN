import numpy as np
import matplotlib.pyplot as plt
import math

class GenerateSampledImage:

    def __init__(self,temp,resolution_length,num_atoms):

        self.temp = temp
        self.resolution_length =  resolution_length
        self.num_atoms = num_atoms
        self.trans_temp = (self.num_atoms/(2*1*1.645))**0.5
        self.temp_ratio = self.temp/self.trans_temp
        self.generated_image = self.generate_image(self.temp,self.temp_ratio,self.resolution_length,self.num_atoms)

    def generate_image(self,temp,temp_ratio,length,num_atoms):
        '''
        generate image using sampling
        '''

        x_list = []
        y_list = []

        for i in range(num_atoms):
            probability = np.random.rand()
            if temp_ratio >1 :
                temp_ratio = 1
            BEC_probability = (1-(temp_ratio)**2)
            if probability <= BEC_probability:
                x_list.append(np.random.normal(0,0.707))
                y_list.append(np.random.normal(0,0.707))
            else:
                x_list.append(np.random.normal(0,math.sqrt(temp)))
                y_list.append(np.random.normal(0,math.sqrt(temp)))



        hist = np.histogram2d(x_list,y_list,100)[0]
        hist = hist.tolist()
        return hist


def main():

    generate = GenerateSampledImage(200,100,100000)
    generate.generate_image(generate.temp,generate.temp_ratio,generate.resolution_length,generate.num_atoms)


if __name__ == "__main__":
    main()
