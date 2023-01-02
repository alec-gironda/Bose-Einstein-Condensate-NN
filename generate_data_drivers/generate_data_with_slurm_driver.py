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

    seed = input()

    GenerateBatch.generate_batch(int(seed))

if __name__ == "__main__":

    main()
