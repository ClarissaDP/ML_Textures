#!/usr/bin/python

import numpy as np
import sys
from sklearn.metrics import confusion_matrix


# *********************** Main program ************************ #
if ( len(sys.argv) < 4 ):
  print("python knn.py train test k [modo]")
  print("* 10 classes")
  sys.exit()


# Open and access the file
tr_f = sys.argv[1]
te_f = sys.argv[2]
k = int(sys.argv[3])


mi = 26
ma = 27

print 'Starting read from Train file...'
temp = np.loadtxt(tr_f, dtype=float, skiprows=1)
train, train_labels, oto = np.hsplit(temp, np.array([mi, ma]))
print 'train.shape = {0}'.format(train.shape)
print 'train_labels.shape = {0}'.format(train_labels.shape)
#print train_labels


print 'Starting read from Test file...'
temp = np.loadtxt(te_f, dtype=float, skiprows=1)
test, test_labels, oto = np.hsplit(temp, np.array([mi, ma]))
print 'test.shape = {0}'.format(test.shape)
print 'test_labels.shape = {0}'.format(test_labels.shape)
#print test_labels

#line,col = test_labels.shape
#test_labels = test_labels.reshape(col,line)


pred = np.array([])
print 'Distancia euclidiana...'
for i in range( len(test) ):

    result = np.array([])

    for j in range( len(train) ):

        sub = np.subtract(test[i], train[j])
        square = np.square(sub) 
        soma = np.sum(square)
        result = np.append(result, soma)
        

    res = np.sqrt(result)
    print 'Test = {0}, Train = {1} - result = {2}'.format(i, j, res.shape)
    # print res
   

    # k melhores
    idx = np.argpartition(res, k)
    best = idx[:k]
    
    #print 'idx:'
    size_best = len(best)    
    melhores = np.array([])
    for h in range( size_best ):
        #print best, res[best[h]] 
        melhores = np.append(melhores, train_labels[best[h]])
        
    topes = melhores.astype(int)
    u, indices = np.unique(topes, return_inverse=True)
    pred = np.append(pred, u[np.argmax(np.bincount(indices))])



save_cm = confusion_matrix(test_labels, pred)

size_cm = len( save_cm )
correct = 0
errado = 0
for i in range( size_cm ):
    for j in range( size_cm ):
        if (i != j):
            errado += save_cm[i][j]
        else:
            correct += save_cm[i][j]


print ''
print 'Output:'
total = correct + errado
print 'Total = {0}, Correto = {1}, Errado = {2}'.format(total, correct, errado)
taxa = float(correct) / float(total)
print 'Taxa de reconhecimento = {0}'.format(taxa)
print save_cm

