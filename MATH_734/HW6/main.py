import numpy
import matplotlib.pyplot as plt

M = 500
tag = range(1,M+1)
target = []

def simu(i, j, c, target, times = 0, stop = M):
    if times >= stop:
        return
    target.append(i/(i+j))
    t = numpy.random.uniform()
    if t <= i/(i+j):
        simu(i+c,j,c,target,times+1)
    else:
        simu(i,j+c,c,target,times+1)
    return

for i in range(15):
    target = []
    simu(2,5,7,target)
    plt.plot(tag,target)

plt.show()