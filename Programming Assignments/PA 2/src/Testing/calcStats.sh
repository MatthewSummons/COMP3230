#!/bin/bash

rslt=results.txt

> $rslt

numThr=(0 1 2 4 6 8 10 12 16 18 20 24 32)
for n in "${numThr[@]}"; do
    filename="./Scores/temp_$n.txt"
    grep -E "* time: [0-9]+\.[0-9]+ s, achieved tok/s: [0-9]+\.[0-9]+" $filename > .hold_spd.txt
    grep -E "main thread -" $filename > .hold_time.txt
    grep -E "Main Thread -" $filename >> .hold_time.txt
    
    echo "$n Threads:" >> results.txt
    python3 calcStats.py >> results.txt
    # echo "$n Threads done!" 
done

echo "$rslt" | python3 organize.py > org_$rslt

rm .hold_spd.txt
rm .hold_time.txt

echo "All Statistics Calculated!"