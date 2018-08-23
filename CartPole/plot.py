from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')
plt.title('Max Fitness in Each Generation')
plt.xlabel('Generations')
plt.ylabel('Fitness')

x = np.load('plots.npy')

xaxis = [i[0] for i in x]
yaxis = [i[1] for i in x]
print(yaxis)
plt.plot(xaxis,yaxis)
plt.show()

