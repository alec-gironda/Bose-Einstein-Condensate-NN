from generate_bec_thermal_cloud_nn_data import GenerateBecThermalCloudData
import pickle

def main():

    training_size = 10000
    test_size = 5000
    noise_spread = 0
    resolution_length = 100
    num_atoms = 100000
    trans_temp = (num_atoms/(2*1*1.645))**0.5

    generate = GenerateBecThermalCloudData(training_size,test_size,noise_spread,resolution_length,num_atoms,trans_temp)

    with open('generated_data.pickle', 'wb') as f:
        pickle.dump(generate, f)


if __name__ == "__main__":

    main()
