import concurrent.futures
import numpy as np
import pickle
import bz2
import time
from generate_data_batches_for_mp import GenerateBatch
import argparse


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

    parser = argparse.ArgumentParser(description="arguments for neural network model")

    parser.add_argument(
    "-t",
    "--train",
    type = int,
    default = 8000,
    help = "size of training data to generate"
    )

    parser.add_argument(
    "-p",
    "--processes",
    type = int,
    default = 8,
    help = "how many processes to run in parallel"
    )

    args = parser.parse_args()

    print("generating data batches...")

    sizes = [args.train//args.processes for i in range(args.processes)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        seeds = np.arange(args.processes)
        executor.map(GenerateBatch.generate_batch,seeds,sizes)

    print("data batches generated. Time for all batches to be generated:")

if __name__ == "__main__":

    main()
