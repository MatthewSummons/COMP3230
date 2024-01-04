#!/bin/bash

programName="llama2_$2"

# Look for and compile the programs
gcc -o seq seq.c utilities.c -O2 -lm
gcc -o $programName llama2_3035946760.c utilities.c -O2 -pthread -lm


# Generate Seeds
numSeeds=$1
seeds=()
for ((i=0; i<numSeeds; i++)); do
    seeds+=($RANDOM)
done

if [[ $2 -le 0 ]]
then
    for s in "${seeds[@]}"; do
        ./seq $s
    done
else
    for s in "${seeds[@]}"; do
        ./$programName $s $2
    done
fi

rm seq
rm $programName