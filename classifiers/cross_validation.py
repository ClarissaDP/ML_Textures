import numpy as np
import sys

from scipy.io import arff
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing

from sklearn import linear_model
from sklearn.lda import LDA
import collections


SIZE = 10


# *********************** Main program ************************ #
if ( len(sys.argv) < 2 ):
  print("python cross_validation.py train [mi] [ma]")
  sys.exit()


# Open and access the file
tr_f = sys.argv[1]


mi = 262
ma = 263

if ( len(sys.argv) > 2 ):
    mi = int(sys.argv[2])
    ma = int(sys.argv[3])


print 'Starting read from Train file...'
temp = np.loadtxt(tr_f, dtype=float, skiprows=1)
print temp.shape
f_train, f_train_labels, oto = np.hsplit(temp, np.array([mi, ma]))
print 'train.shape = {0}'.format(f_train.shape)
print 'train_labels.shape = {0}'.format(f_train_labels.shape)


ini,l = f_train.shape 
prop = int ((40 * ini)/100)


dt = np.array([])
gnb = np.array([])
mnb = np.array([])
bnb = np.array([])
knn = np.array([])
sv = np.array([])
per = np.array([])
ldan = np.array([])
lr_ovr = np.array([])
lr_mul = np.array([])

knn_h = {}
k_range = [5, 9, 11, 13]
for i in k_range:
    knn_h[i] = 0.0



def GridSearch(X_train, y_train):

        # define range dos parametros
        C_range = 2. ** np.arange(-5,15,2)
        gamma_range = 2. ** np.arange(3,-15,-2)
        k = [ 'rbf']
        #k = ['linear', 'rbf']
        param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

        # instancia o classificador, gerando probabilidades
        srv = svm.SVC(probability=True)

        # faz a busca
        grid = GridSearchCV(srv, param_grid, n_jobs=-1, verbose=True)
        print "grid.fit"
        grid.fit (X_train, (y_train.ravel()) )

        # recupera o melhor modelo
        model = grid.best_estimator_

        # imprime os parametros desse modelo
        print grid.best_params_
        print model
        return model
        

# Validacao cruzada 10 vezes - prop: 60/40
for i in range(0, SIZE):
    
    train = f_train
    train_labels = f_train_labels
    test = np.array([])
    test_labels = np.array([])
   
    for j in range(0, prop):
        a = train.shape[0] - 1
        rand = random.randint(0, a)
        
        test = np.append(test, train[rand])
        test_labels = np.append(test_labels, train_labels[rand])
        
        train = np.delete(train, rand, 0)
        train_labels = np.delete(train_labels, rand, 0)
    '''
    train, test, train_labels, test_labels = train_test_split(f_train, f_train_labels, test_size=0.40, random_state=42)
    '''
    test = np.reshape(test, ((ini-train.shape[0]), l))
    print train.shape, test.shape 
    
    tr_labels = train_labels.ravel()
    te_labels = test_labels.ravel()


    # ********************* Funcoes ********************** #
    
    # Decision Tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train, train_labels)

    print ''
    print '******************************************************'
    print 'Decision Tree:'

    pred = np.array([])
    pred = clf.predict(test)
    
    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix(pred, test_labels)
    dt = np.append(dt, acc)

    # Gaussian Naive Bayes
    print ''
    print '******************************************************'
    print 'Gaussian Naive Bayes:'

    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)
    pred = np.array([])
    clf = GaussianNB()
    pred = clf.fit(train, train_labels).predict(test)
    
    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    gnb = np.append(gnb, acc)

    '''
    # Multinomial Naive Bayes
    print ''
    print '******************************************************'
    print 'Multinomial Naive Bayes:'

    pred = np.array([])
    clf = MultinomialNB()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    mnb = np.append(mnb, acc)


    # Bernoulli Naive Bayes
    print ''
    print '******************************************************'
    print 'Bernoulli Naive Bayes:'

    pred = np.array([])
    clf = BernoulliNB()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    bnb = np.append(bnb, acc)
    '''
    
    # KNN
    print ''
    print '******************************************************'
    print 'KNN:'


    for k in k_range:
        
        print '---------------------'
        print 'KNN == ', k, ':'
        
        pred = np.array([])
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train, train_labels) 
        pred = np.append(pred, (neigh.predict( test )) )

        acc = accuracy_score(pred, test_labels)
        print ''
        print acc
        print confusion_matrix( test_labels, pred )
        knn = np.append(knn, acc)
        knn_h[k] += acc
            
    
    # SVM
    print ''
    print '******************************************************'
    print 'SVM:'

    # CI171Train
    # LBP, GLCM e PCA
    #best = svm.SVC(C=8192.0, gamma=8.0, kernel='rbf', probability=True)
    # qual ??
    #best = svm.SVC(C=8192.0, gamma=0.0001220703125, kernel='rbf', probability=True)
    # Histograma de cor
    best = svm.SVC(C=0.03125, gamma=8.0,kernel='rbf', probability=True)

    # Softwood
    #best = svm.SVC(C=8192.0, gamma=0.03125, kernel='rbf', probability=True)
    #best = svm.SVC(C=2048.0, gamma=0.125, kernel='rbf', probability=True)
    
    best.fit(train, tr_labels)
    acc = best.score(test, test_labels)
    
    print ''
    print acc
    #print confusion_matrix( test_labels, pred )
    sv = np.append(sv, acc)


    # Perceptron
    print ''
    print '******************************************************'
    print 'Perceptron:'

    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
    pred = np.array([])
    clf = linear_model.Perceptron()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    per = np.append(per, acc)
    

    # LDA
    print ''
    print '******************************************************'
    print 'LDA:'

    # http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
    pred = np.array([])
    clf = LDA()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    ldan = np.append(ldan, acc)


    # Logistic Regression - ovr
    print ''
    print '******************************************************'
    print 'Logistic Regression - ovr:'

    pred = np.array([])
    clf = linear_model.LogisticRegression()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    lr_ovr = np.append(lr_ovr, acc)


    # Logistic Regression - multinomial
    print ''
    print '******************************************************'
    print 'Logistic Regression - mul:'

    pred = np.array([])
    clf = linear_model.LogisticRegression( solver='newton-cg', multi_class='multinomial' )
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    lr_mul = np.append(lr_mul, acc)
    
    print '######################################################'


print '######################################################'

print 'DT: '
print dt
print np.mean(dt)

print 'gnb: '
print gnb
print np.mean(gnb)
'''
print 'mnb: '
print mnb
print np.mean(mnb)

print 'bnb: '
print bnb
print np.mean(bnb)
'''
print 'knn: '
print knn
print np.mean(knn)
knn_h_ordered = collections.OrderedDict( knn_h )
for key in ( knn_h_ordered ):
    print key, ' = ', (knn_h_ordered[key] / SIZE)


print 'svm: '
print sv
print np.mean(sv)

print 'per: '
print per
print np.mean(per)

print 'lda: '
print ldan
print np.mean(ldan)

print 'lr_ovr: '
print lr_ovr
print np.mean(lr_ovr)

print 'lr_mul: '
print lr_mul
print np.mean(lr_mul)

