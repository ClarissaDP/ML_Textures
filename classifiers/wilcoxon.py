#!/usr/bin/python

import numpy as np
import sys
from sklearn.metrics import confusion_matrix

from scipy.stats import wilcoxon



# *********************** Main program ************************ #
if ( len(sys.argv) < 2 ):
  print("python wilcoxon.py result")
  sys.exit()


# Open and access the file
f = sys.argv[1]

f = open(f)
data = f.readlines()
f.close()

nm = np.array([])
fl = np.array([])
size = len(data)
for i in range( size ):
    temp = np.array([])
    data[i] = data[i].rstrip('\n')
    if ( i % 2 == 0 ):
        nm = np.append( nm, data[i] )
    else:
        temp = np.append( temp, data[i].split(' ') )
        for i in range(1, len(temp)):
            fl = np.append( fl, temp[i] )
            



for i in range( len(nm) ):
    print nm[i]
    print fl[i]

