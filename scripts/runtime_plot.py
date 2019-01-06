# this is not a ros node
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log_0 = np.array(pd.read_csv('log_00.txt')[:1000])
log_1 = np.array(pd.read_csv('log_01.txt')[:1000])
fps30 = np.full((1000,1),1/30.0)
fps60 = fps30/2
fps90 = fps30/3

y_max = max(np.vstack((log_0,log_1,fps30))) * 1.05
# y_max = max(log_1) * 1.05

plt.plot(log_0,label='decode_runtime')
plt.plot(log_1,label='current_node_runtime')
plt.plot(fps30,label='30Hz')
plt.plot(fps60,label='60Hz')
plt.plot(fps90,label='90Hz')

plt.legend()

# plt.plot(xs,log_0)
# plt.plot(xs,log_1)
# plt.plot(xs,f)
plt.xlim([0,999])
plt.ylim([0,y_max])

plt.show()
