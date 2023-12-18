#!/bin/bash

mkdir -p logs

# Loop from 0 to 9
for i in {0..9}
do
   echo "Starting process with argument: $i"
   python -u generate_influential_queries.py $i > logs/generate_influential_queries_$i.txt 2>&1 &
done
