# Testing The Program

The testing for this program is slightly complicated because if we try to automate everything we get *different* results than what we would get otherwise because displaying text to STDOUT significantly slows down the program. I hit ~700 tok/s if I redirected the output into a file for example, and it seemed like no matter how many threads I tried with I got the same result, as if something else was being the bottleneck.

However to ensure that the results were comparable I wrote a bash script (with *some* help from ChatGPT) called test_ll2.sh which takes two arguments, the number of random seeds you want to test with and the number of threads to test with. This script must be in the same directory as seq.c and llama2_[UIID].c to work. It also works with 0 threads as an arrgument. An example of this would be:

```
# Test the program with 12 threads using 100 different seeds
# Repeat for n = (1, 2, 4, ...) and save output in ./Scores/temp_n.txt
./test_ll2.sh 100 12 
```

This will print the output to the screen (as needed to match "reality").

I then copy the output on the screen into the file temp_12.txt (indicating that this file holds the data for execution with 12 threads) in the Scores directory. Rinse and repeat for however many different threads or seeds you want.

After this I calculate the average time, speed etc. by first parsing and then averaging the data using calcStats.sh (which uses calcStats.py to help with the calculation and organize.py for *organization* of the data). Run as follows:

```
./calcStats.sh
```

This will produce two text files, results.txt and org_results.txt. We use results.txt to fill the table for the assignment and org_results.txt to plot the results (using graph.py). 

```
# Requires matplotlib to run
python3 graph.py
```

It should be noted that my laptop could not handle running another OS and so benchmarking was performed on the CS Server (TeeHee).
