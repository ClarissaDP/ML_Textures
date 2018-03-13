#!/usr/bin/python

# Usando sklearn, teste as diferentes variantes disponiveis para o NayveBayes e compare os resultados com o KNN e DT.


import numpy as np
import sys
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB



# *******
def print_confusionMatrix( test_labels, pred ):
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



# *********************** Main program ************************ #
if ( len(sys.argv) < 3 ):
  print("python nayveBayes.py train test")
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

#line,col = test_labels.shape
#test_labels = test_labels.reshape(col,line)


train_labels = np.ravel(train_labels)
test_labels = np.ravel(test_labels)

# Gaussian Naive Bayes
print ''
print '******************************************************'
print 'Gaussian Naive Bayes:'

pred = np.array([])
gnb = GaussianNB()
pred = gnb.fit(train, train_labels).predict(test)
#print("Number of mislabeled points out of a total %d points : %d" 
#		% (test.shape[0],(test_labels != pred).sum()))


print_confusionMatrix( test_labels, pred )


# Multinomial Naive Bayes
print ''
print '******************************************************'
print 'Multinomial Naive Bayes:'

pred = np.array([])
clf = MultinomialNB()
clf.fit(train, train_labels)
pred = clf.predict(test)

print_confusionMatrix( test_labels, pred )


# Bernoulli Naive Bayes
print ''
print '******************************************************'
print 'Bernoulli Naive Bayes:'

pred = np.array([])
clf = BernoulliNB()
clf.fit(train, train_labels)
pred = clf.predict(test)

print_confusionMatrix( test_labels, pred )
 


# Partial Fit
#	training data might not fit in memory. 

# Gaussian Naive Bayes (Partial Fit)
#	clf_pf = GaussianNB()
#	clf_pf.partial_fit(train_labels, train_labels, np.unique(train_labels))
#	pred = clf_pf.predict(test)
