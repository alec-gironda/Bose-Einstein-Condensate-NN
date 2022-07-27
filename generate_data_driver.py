from generate_temp_nn_data import GenerateData
import csv
import msgspec

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

def main():

    training_size = 1000
    test_size = 500
    noise_spread = 0.03
    resolution_length = 100

    data = GenerateData(training_size,test_size,noise_spread,resolution_length).data_list
    data.append(resolution_length)
    data = tuple(data)
    jsonObj = msgspec.json.encode(data)
    with open('temp_nn_data.json', 'wb') as f:
        f.write(jsonObj)

if __name__ == "__main__":

    main()
