#!/bin/bash

if [ $# -ne 3 ]; then
  >&2 echo "Usage: $0 input_dir output_dir csv_file"
  exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
CSV_FILE=$3

DIR=$(dirname "$(readlink -f "$BASH_SOURCE")")

set -x
set -e

cd $DIR
export PYTHONPATH=$PYTHONPATH:$DIR

echo "Workdir is $DIR. Contents are:"
ls $DIR

echo "Running mrqy/QC.py"
python mrqy/QC.py ${OUTPUT_DIR} ${INPUT_DIR}/file_data

echo "Setting up output"
cp ${OUTPUT_DIR}/UserInterface/Data/IQM.csv ${OUTPUT_DIR}/cohort_data.csv
