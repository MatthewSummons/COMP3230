# Receives summarised/organised data from data.pickle and plots it

import pickle
import matplotlib.pyplot as plt

with open("L.pickle", "rb") as file:
    loaded_data = pickle.load(file)

print(loaded_data)

x = loaded_data["Threads"]

plt.figure(1, figsize=(10, 7))
plt.subplot(121)
plt.plot(x, loaded_data["Average Time"], "bo", label="Average Time")
plt.xlabel("Threads")
plt.ylabel("Average Time (seconds s)")
plt.title("Waiting Time", pad=20)

plt.subplot(122)
plt.plot(x, loaded_data["Average Achieved Toks/s"], "bo", label="Average Achieved Toks/s")
plt.xlabel("Threads")
plt.ylabel("Average Speed (tok/s)")
plt.title("Speed", pad=20)

plt.subplots_adjust(left=0.075, right=0.95, top=0.825, wspace=0.3)
plt.suptitle("Performance I", fontsize=24)

plt.figure(2, figsize=(12, 8))

plt.subplot(131)
plt.plot(x, loaded_data["Average User Time"], "ko", label="Average User Time")
plt.xlabel("Threads")
plt.ylabel("Average User Time (seconds s)")
plt.title("User Time", pad=20)

plt.subplot(132)
plt.plot(x[:-1], loaded_data["Average System Time"][:-1], "ko", label="Average System Time")
plt.xlabel("Threads")
plt.ylabel("Average System Time (seconds s)")
plt.title("System Time", pad=20)

plt.subplot(133)
plt.plot(x[1:], loaded_data["User/System Time Ratio"][1:], "ro", label="Use Time/System Time")
plt.xlabel("Threads")
plt.ylabel("Ratio")
plt.title("User Time/System Time", pad=20)

plt.subplots_adjust(left=0.075, right=0.95, top=0.825, wspace=0.3)
plt.suptitle("Performance II", fontsize=24)

plt.show()