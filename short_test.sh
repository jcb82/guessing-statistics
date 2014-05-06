#!/bin/sh

test_dir=test_dists

# clean up a raw text file
sed -r 's/(.)/\1\n/g' < ${test_dir}/un_udhr_raw.txt |\
tr '[:upper:]' '[:lower:]' |\
bash reduce.sh > ${test_dir}/un_udhr.txt

# convert from text to compressed representation
python compress.py ${test_dir}/un_udhr.txt ${test_dir}/ry_pins.txt

# print basic statistics
python testStats.py ${test_dir}/un_udhr.dz ${test_dir}/ry_pins.dz

# make a basic plot
python testPlot.py ${test_dir}/un_udhr.dz ${test_dir}/ry_pins.dz
