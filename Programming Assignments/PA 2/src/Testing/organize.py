'''
Given the results for each thread, organise and collect them for graphing
'''

import pickle

# Collect and organize the data to graph
threads = []
average_time_list, achieved_toks_list = [], []
average_user_time_list, average_system_time_list, user_system_ratio_list = [], [], []

with open(input(), "r") as file:
    lines = file.readlines()

    i = 0
    while i < len(lines):
        if "Threads" in lines[i]:
            thrNum = int(lines[i].split()[0])
            threads.append(thrNum) 
        elif "Average Time" in lines[i]:
            average_time = float(lines[i].split(": ")[1].split(" ")[0])
            average_time_list.append(average_time)
        elif "Average Achieved Toks/s" in lines[i]:
            achieved_toks = float(lines[i].split(": ")[1].split("\n")[0])
            achieved_toks_list.append(achieved_toks)
        elif "Average User Time" in lines[i]:
            average_user_time = float(lines[i].split(": ")[1].split(" ")[0])
            average_user_time_list.append(average_user_time)
        elif "Average System Time" in lines[i]:
            average_system_time = float(lines[i].split(": ")[1].split(" ")[0])
            average_system_time_list.append(average_system_time)
        elif "User/System Time Ratio" in lines[i]:
            user_system_ratio = float(lines[i].split(": ")[1].split("\n")[0])
            user_system_ratio_list.append(user_system_ratio)
        
        i += 1


collection = {
    "Threads": threads,
    "Average Time": average_time_list,
    "Average Achieved Toks/s": achieved_toks_list,
    "Average User Time": average_user_time_list,
    "Average System Time": average_system_time_list,
    "User/System Time Ratio": user_system_ratio_list
}

for key, value in collection.items():
    print(key, end=": ")
    for t in value:
        print(t, end=" ")
    print()

with open("data.pickle", "wb") as file:
    pickle.dump(collection, file)