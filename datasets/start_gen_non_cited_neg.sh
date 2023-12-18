#!/bin/bash

mkdir -p logs

for i in {0..9}
do
   echo "Starting process with argument: $i"
   python -u gen_trips_non_cited_neg.py $i > logs/gen_trips_non_cited_neg_$i.txt 2>&1 &
done
