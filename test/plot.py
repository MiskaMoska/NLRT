import pickle
import matplotlib 
from matplotlib import pyplot as plt

indices = [1, 2, 3, 4]

ax = plt.figure(figsize=(15,10))
colors = ['blue', 'red', 'purple']
labels = ['Real SA (Accept worse solutions with a probability', 'Dummy SA (Never accept worse solutions)', 'Random (No optimization algorithm is used)']
file_base = ['real', 'dummy', 'rand']
colors.reverse()
labels.reverse()
file_base.reverse()

for j, b in enumerate(file_base):
    for i in indices:
        file_name = f"data/{b}_{i}.pkl"
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            data = data[:1000]

        plt.plot(
            [i/1000 for i in range(len(data))], 
            data, 
            label=labels[j] if i == 1 else None, 
            color=colors[j], 
            linewidth=2
        )

plt.xlabel("Normalized Running Time", fontsize=15)
plt.ylabel("Objective Function Value", fontsize=15)
plt.legend(fontsize=15)
plt.show()




