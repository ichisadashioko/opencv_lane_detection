# this is not a ros node
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log_0 = np.array(pd.read_csv('log_00.txt')[:1000])
log_1 = np.array(pd.read_csv('log_01.txt')[:1000])
f = np.full((1000,1),1/60.0) # 60 Hz

print('log_0: {}, log_1: {}, f: {}'.format(log_0.shape,log_1.shape,f.shape))
y_max = max(np.vstack((log_0,log_1,f))) * 1.05
# y_max = max(log_1) * 1.05

plt.plot(log_0,label='none-process')
plt.plot(log_1,label='process')

plt.legend()

# plt.plot(xs,log_0)
# plt.plot(xs,log_1)
# plt.plot(xs,f)
plt.xlim([0,999])
plt.ylim([0,y_max])

plt.show()