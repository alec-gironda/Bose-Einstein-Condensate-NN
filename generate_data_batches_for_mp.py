from generate_bec_thermal_cloud_nn_data import GenerateBecThermalCloudData
import pickle
import bz2
import time
import pathlib

class GenerateBatch:

    '''

    this driver script allows the user to generate images of BECs and their surrounding
    thermal cloud using a probability sampling technique and save them as a compressed, serialized
    bz2 file.

    data generation parameters:

        training_size (int) : number of images to generate for training a model
        test_size (int) : number of images to generate for testing a model
        noise_spread (float) : larger spread adds more noise to the image, but not fully implemented yet right now, so default to 0
        resolution_length (int) : generated images will be of size resolution_length * resolution_length pixels
        num_atoms (int) : total number of atoms to be generated in the images
        dimensions (int) : generate images based on a 2D or 3D model

    '''

    def generate_batch(seed_value):
        '''

        generates a batch of images

        '''

        #set parameters

        training_size = 1000
        test_size = 500
        noise_spread = 0
        resolution_length = 100
        num_atoms = 100000
        dimensions = 2
        seed = seed_value

        if dimensions == 2:
            trans_temp = (num_atoms/(2*1*1.645))**(1/dimensions)
        elif dimensions == 3:
            trans_temp = (num_atoms/(2*2*1.202))**(1/dimensions)

        #generate images based on parameters

        generate = GenerateBecThermalCloudData(training_size,test_size,noise_spread,resolution_length,num_atoms,trans_temp,dimensions,seed)

        #write generated images to bz2 file

        cwd = pathlib.Path(__file__).parent.resolve()
        o_file = str(cwd) + "/generated_data/generated_data" + str(seed_value) + ".bz2"
        out = bz2.BZ2File(o_file,'wb')
        pickle.dump(generate,out)
        out.close()
