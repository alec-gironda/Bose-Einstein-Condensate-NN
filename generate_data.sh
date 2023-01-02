#!/bin/bash

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                show help"
      echo "-s, --slurm               multiprocess using SLURM"
      exit 0
      ;;

    -s|--slurm)

      sbatch ./generate_data_drivers/slurm_script.sh
      python3 ./generate_data_drivers/combine_data_batches_into_full_data.py

      exit 0

      ;;

    *)

      echo "Flag not recognized. Use the flag -h or --help for help."

      exit 1

    ;;

  esac

done

python3 ./generate_data_drivers/generate_data_no_slurm_driver.py
python3 ./generate_data_drivers/combine_data_batches_into_full_data.py
