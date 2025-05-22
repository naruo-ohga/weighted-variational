import numpy as np
import matplotlib.pyplot as plt

# Read file
filename = 'weighted_variational_test_extremum.txt'
with open(filename) as f:
    lines = f.readlines()
    
opt = float(lines[1].split()[0])

data = []
for line in lines[4:]:
    data.append([float(x) for x in line.split()])

data = np.array(data)


# Plot
x = data[:,0]
fx = data[:,1]
gx = data[:,2]

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()

ax.plot(x, fx, label='f(x)')
ax2.plot(x, gx, c='orange', label='g(x)')
ax.axvline(x=opt, color='r', linestyle='--', label='opt')
ax2.axhline(y=0, color='k', linestyle='--')

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax2.set_ylabel('g(x)')

ax2.set_ylim(np.min(gx) * 1.1, -np.min(gx) * 1.5)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2)

plt.show()