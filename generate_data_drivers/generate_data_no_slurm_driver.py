import concurrent.futures
import numpy as np
import pickle
import bz2
import time
from generate_data_batches_for_mp import GenerateBatch


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
def main():

    print("generating data batches...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        seeds = np.arange(8)
        executor.map(GenerateBatch.generate_batch,seeds)

    print("data batches generated.")
    
if __name__ == "__main__":

    main()
