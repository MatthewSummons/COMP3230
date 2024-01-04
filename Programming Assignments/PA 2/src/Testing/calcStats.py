'''
Given Scores--The Test Data for a single thread, parse the output and calculate the mean statistics for selected fields
Please run calcStats.sh
'''

spd_filename = ".hold_spd.txt"
time_filename = ".hold_time.txt"
total_time, total_toks, count_s = 0, 0, 0
total_user_time, total_sys_time, count_t = 0, 0, 0

# Calculate mean time and token speed
with open(spd_filename, "r") as file:
    for line in file:
        if (line.startswith("length:")) and ("time:" in line) and ("achieved tok/s:" in line):
            line = line.strip()
            time = float(line.split(",")[1].split(":")[1].strip().split()[0])
            achieved_toks = float(line.split(",")[2].split(":")[1].strip())
            
            total_time += time
            total_toks += achieved_toks
            count_s += 1

# Calculate mean user and system time
with open(time_filename, "r") as time_file:
    for line in time_file:
        if line.lower().startswith("main thread - user:") and ("system:" in line):
            line = line.strip()
            user_time = float(line.split("user:")[1].split("s")[0].strip())
            system_time = float(line.split("system:")[1].split("s")[0].strip())
            
            total_user_time += user_time
            total_sys_time += system_time
            count_t += 1

assert count_s == count_t, f"{count_s=}, {count_t=}"

if count_s > 0:
    average_time = total_time / count_s
    average_toks = total_toks / count_s
    
    print(f"Average Time: {average_time:.3f} s")
    print(f"Average Achieved Toks/s: {average_toks:3f}")

    average_user_time = total_user_time / count_t
    average_sys_time = total_sys_time / count_t

    print(f"Average User Time: {average_user_time:.3f} s")
    print(f"Average System Time: {average_sys_time:.3f} s")
    print(f"User/System Time Ratio: {average_user_time / average_sys_time:.3f}")
else:
    print("No valid lines found in the file.")