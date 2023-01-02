#!/bin/bash

while getopts h:t:p:s: flag
do
    case "${flag}" in
      h)
        echo "options:"
        echo "-h                show help"
        echo "-s                multiprocess using SLURM"
        echo "-t                size of training data. test data will be half the size."
        echo "-p                number of processes to generate data with"
        exit 0
        ;;

      t)

        train_size=${OPTARG}

        ;;

      p)

        num_processes=${OPTARG}

        ;;

      s)

        sbatch ./generate_data_drivers/slurm_script.sh
        python3 ./generate_data_drivers/combine_data_batches_into_full_data.py

        exit 0

        ;;

      *)

        echo "Flag not recognized or missing argument. Use the flag -h or --help for help."

        exit 1

      ;;

  esac

done

for i in "$train_size","-t" "$num_processes","-p";
do
  IFS=",";
  set -- $i;
  if [ -z $1 ] ;
  then
    echo "missing required argument: $2"
    exit 1
  fi
done

python3 ./generate_data_drivers/generate_data_no_slurm_driver.py -t $train_size -p $num_processes
python3 ./generate_data_drivers/combine_data_batches_into_full_data.py
