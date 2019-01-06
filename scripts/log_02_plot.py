from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

log_02 = pd.read_csv('log_02.txt')

log_array = np.array(log_02)

start_time,	pre_process,bird_view,sliding_windows,draw_lane,run_time = np.transpose(log_array)

total_runtime = run_time - start_time

plt.plot(pre_process,label='pre_process')
plt.plot(bird_view,label='bird_view')
plt.plot(draw_lane,label='draw_lane')
plt.plot(total_runtime,label='runtime')
plt.legend()
plt.show()


# array_split= np.split(log_array,6)