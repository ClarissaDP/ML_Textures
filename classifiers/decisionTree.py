#!/usr/bin/python
#
# Exercicio 1
# - Utilize a implementacao do scikit-learn e classifique as bases de dados
#   fornecidas para o knn para treinar e testar a arvore de decisao.
# - Teste os diferentes parametros do classificador e analise os resultados.
# - Reporte a configuracao que obteve a melhor acuracia.

import numpy as np
import sys
#from skmultilearn.dataset import Dataset
from scipy.io import arff
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# *********************** Main program ************************ #
if ( len(sys.argv) < 3 ):
  print("python decisionTree.py train test")
  sys.exit()


# Open and access the file
tr_f = sys.argv[1]
te_f = sys.argv[2]


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



# criar arvore de decisao
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train, train_labels)

# classificar...
print ''
#print 'Classificado:'
pred = clf.predict(test)

# confusion matrix and accuracy
#   Ver knn_numpy se quiser toda a verificacao
save_cm = confusion_matrix(test_labels, pred)
print save_cm

print 'Taxa de reconhecimento (funcao):'
best_accuracy = accuracy_score(pred, test_labels)
print best_accuracy


# Testes de parametros
print ''
print 'Testes:'

clf = clf.fit(train, train_labels)
pred = clf.predict(test)

acc = accuracy_score(pred, test_labels)
print acc
print confusion_matrix(pred, test_labels)


'''
file_name = "arvore.dot"
with open(file_name, 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

# dot -Tpdf arvore.dot -o arvore.pdf


feature_names = ['outlook', 'temperature', 'humidity', 'windy']
target_names = ['yes', 'no']

oto_file_name = "ota_arvore.dot"
with open(oto_file_name, 'w') as f:
	f = tree.export_graphviz(clf, out_file=f,  
							feature_names=feature_names,  
							class_names=target_names,  
							filled=True, rounded=True,  
							special_characters=True) 
'''
