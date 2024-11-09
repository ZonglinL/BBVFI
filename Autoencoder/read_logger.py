from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


acc = EventAccumulator(
"/scratch/zl3958/VLPR/BBVFI/logs/2024-10-10T23-02-36_vqflow-f32/testtube/version_0/events.out.tfevents.1728616067.gv004.hpc.nyu.edu.1572102.0"
)
acc.Reload()
loss = 'train/rec_loss_epoch'
val_loss = 'val/rec_loss'
'''
loss = 'train/g_loss_epoch'
val_loss = 'val/g_loss'
'''
print(acc.Tags())

df = pd.DataFrame(acc.Scalars(loss)).value
train = df

df = pd.DataFrame(acc.Scalars(val_loss)).value
test = df
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(train)
ax.plot(test)
ax.legend(['train loss','val loss'])
fig.savefig('loss.png')   # save the figure to file
plt.close(fig) 