import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import torch.nn as nn


Real = np.load('/Users/yangkaisen/MyProject/EEG-BCI/results/Real.npy')
Pred = np.load('/Users/yangkaisen/MyProject/EEG-BCI/results/Pred.npy')
me = np.loadtxt('/Users/yangkaisen/MyProject/EEG-BCI/results/metrics.csv')

import matplotlib.pyplot as plt
import numpy as np


# draw1:  pred results
x = [i for i in range(len(Pred))]
y_labels = ['Left', 'Right', 'Foot', 'Tongue','-']

plt.figure(figsize=(12, 3))
plt.scatter(x, Real+0.03, c='black', s=3, alpha=0.8, label = 'Real', marker='^')
plt.scatter(x, Pred-0.03, c='blue',s=3, alpha= 0.8 ,label = 'Pred', marker='v')
plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
plt.legend(ncol=2)
plt.savefig(fname= './results/trails.png', dpi=500, format='png',bbox_inches='tight')
plt.show()
print(';')


# draw2:  pred results 2
x = [i for i in range(len(Pred))]
y_labels = ['Left', 'Right', 'Foot', 'Tongue','-']
correct,error = [],[]
for t in x:
    if Pred[t] == Real[t]:
        correct.append([t,Pred[t]])
    else:
        error.append([t,Pred[t]])
correct = np.array(correct)
error = np.array(error)

plt.figure(figsize=(12, 3))
plt.scatter(x, Real+0.03, c='black', s=3, alpha=0.8, label = 'Real', marker='^')
plt.scatter(x, Pred-0.03, c='blue',s=3, alpha= 0.8 ,label = 'Pred', marker='v')
plt.vlines(x=correct[:,0], ymin=correct[:,1]-0.3, ymax=correct[:,1]+0.3,color='green', alpha=0.6, linewidth=1, label = 'correct')
plt.vlines(x=error[:,0], ymin=error[:,1]-0.15, ymax=error[:,1]+0.15,color='red', alpha=0.6, linewidth=1, label = 'error')
plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
plt.legend(ncol=4)
plt.savefig(fname= './results/trails_correct.png', dpi=500, format='png',bbox_inches='tight')
plt.show()
print(';')




# draw2:  metrics
plt.plot(me[:,0],me[:,1], color = 'red', label = 'Accuracy')
plt.plot(me[:,0],me[:,2],color = 'green',label = 'Kappa')
plt.plot(me[:,0],me[:,3], color = 'blue', label = 'F1-Score')
plt.legend()
plt.grid(True)
plt.savefig(fname= './results/metrics.png', dpi=500, format='png',bbox_inches='tight')
plt.show()

print(';')