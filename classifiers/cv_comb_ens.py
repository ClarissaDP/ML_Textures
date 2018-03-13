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


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import collections


from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier


SIZE = 10


# Tuto:
# http://scikit-learn.org/stable/supervised_learning.html *** 

# Combinacao??
# http://scikit-learn.org/stable/modules/pipeline.html
#       -> nem...

# Ensembles?
# http://scikit-learn.org/stable/modules/ensemble.html




    
# ********************* Funcoes ********************** #
def decision_tree(train, train_labels, test):   
    # Decision Tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train, train_labels)

    print ''
    print '----------------'
    print 'Decision Tree:'

    pred = np.array([])
    pred = clf.predict(test)
    
    return pred


def bayes(train, train_labels, test):
    # Gaussian Naive Bayes
    print ''
    print '----------------'
    print 'Gaussian Naive Bayes:'

    clf = GaussianNB()
    pred = clf.fit(train, train_labels).predict(test)

    return pred


def knn_f(train, train_labels, test):  
    # KNN
    print ''
    print '----------------'
    print 'KNN:'
    
    k = 5    
        
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train, train_labels) 
    pred = (neigh.predict( test ))

    return k


def svm_f(train, train_labels, test):
    # SVM
    print ''
    print '----------------'
    print 'SVM:'

    #best = svm.SVC(C=8192.0, gamma=8.0, kernel='rbf', probability=True)
    best = svm.SVC(C=0.03125, gamma=8.0,kernel='rbf', probability=True)
    
    #best = svm.SVC(C=8192.0, gamma=0.03125, kernel='rbf', probability=True)
    best = best.fit(train, train_labels)
    #acc = best.score(test, test_labels)
    pred = best.predict(test)
    
    #print pred
    return pred


def perceptron_f(train, train_labels, test):
    # Perceptron
    print ''
    print '----------------'
    print 'Perceptron:'

    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
    pred = np.array([])
    clf = linear_model.Perceptron()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    return pred


def lda_f(train, train_labels, test):
    # LDA
    print ''
    print '----------------'
    print 'LDA:'

    # http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
    clf = LDA()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    return pred


def logistic_ovr(train, train_labels, test):
    # Logistic Regression - ovr
    print ''
    print '----------------'
    print 'Logistic Regression - ovr:'

    pred = np.array([])
    clf = linear_model.LogisticRegression()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    return pred


def logistic_multi(train, train_labels, test):
    # Logistic Regression - multinomial
    print ''
    print '----------------'
    print 'Logistic Regression - mul:'

    pred = np.array([])
    clf = linear_model.LogisticRegression( solver='newton-cg', multi_class='multinomial' )
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    return pred


def acha_maioria( pred ):
    print 'combe: '
    tam = len(pred)
    tam_train = len(test)
    maioria = np.array([])
    
    for i in range(0, tam_train):
        todos = pred[0:tam, i]
        melhor = (collections.Counter(todos).most_common()[0][0] )
        maioria = np.append( maioria, melhor )
    
    return maioria


def printae(maioria, test_labels):
    acc = accuracy_score(maioria, test_labels)
    print ''
    print acc
    print confusion_matrix(maioria, test_labels)

    return acc


# *********************** Main program ************************ #
if ( len(sys.argv) < 2 ):
  print("python cv_comb_ens.py train [mi] [ma]")
  sys.exit()


# Open and access the file
tr_f = sys.argv[1]


mi = 120
ma = 121

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

comb = np.zeros([SIZE, SIZE])
ens = np.zeros([SIZE, SIZE])

k = 0
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
    
    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()


    
    # ********************* Combinacoes ********************** #
    print ''
    print '******************************************************'
    print 'Combinacoes:'
    
    print '### lda + svm ### '
    tam_train = len(test)
    pred = np.zeros([3, tam_train])

    pred[0] = (lda_f(train, train_labels, test))
    pred[1] = (svm_f(train, train_labels, test))

    maioria = acha_maioria(pred)
    acc = printae(maioria, test_labels)    
    comb[0,k] = (acc)
    
    # otas
    print '### knn + lda + svm ### '
    tam_train = len(test)
    pred = np.zeros([3, tam_train])

    pred[0] = (knn_f(train, train_labels, test))  
    pred[1] = (lda_f(train, train_labels, test))
    pred[2] = (svm_f(train, train_labels, test))

    maioria = acha_maioria(pred)
    acc = printae(maioria, test_labels)    
    comb[1,k] = (acc)

    # mais combinacoes
    print '------------------------------'
    print '### log_mul + svm + bayes ### '
    tam_train = len(test)
    pred = np.zeros([3, tam_train])

    pred[0] = (logistic_multi(train, train_labels, test))
    pred[1] = (svm_f(train, train_labels, test))
    pred[2] = (bayes(train, train_labels, test))

    maioria = acha_maioria(pred)
    acc = printae(maioria, test_labels)    
    comb[2,k] = (acc)

    # mais combinacoes
    print '------------------------------'
    print '### all ### '
    tam_train = len(test)
    pred = np.zeros([8, tam_train])

    pred[0] = (decision_tree(train, train_labels, test))
    pred[1] = (bayes(train, train_labels, test))
    pred[2] = (knn_f(train, train_labels, test))  
    pred[3] = (svm_f(train, train_labels, test))
    pred[4] = (perceptron_f(train, train_labels, test))
    pred[5] = (lda_f(train, train_labels, test))
    pred[6] = (logistic_ovr(train, train_labels, test))
    pred[7] = (logistic_multi(train, train_labels, test))
    
    maioria = acha_maioria(pred)
    acc = printae(maioria, test_labels)    
    comb[3,k] = (acc)
    
    
    # ********************* Ensembles ********************** #
    print ''
    print '******************************************************'
    print 'Ensembles:'
   
    print ''
    print '### DecisionTreeClassifier ###'
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    clf = clf.fit(train, train_labels)
    pred = clf.predict(test)
    
    scores = cross_val_score(clf, train, train_labels)
    print 'score.mean(): ', scores.mean()       
    acc = printae(pred, test_labels)    
    ens[0,k] = (acc)
    
    
    print ''
    print '### RandomForestClassifier ###'
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf = clf.fit(train, train_labels)
    pred = clf.predict(test)
    
    scores = cross_val_score(clf, train, train_labels)
    print 'score.mean(): ', scores.mean()       
    acc = printae(pred, test_labels)    
    ens[1,k] = (acc)
    
    
    print ''
    print '### ExtraTreesClassifier ###'
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf = clf.fit(train, train_labels)
    pred = clf.predict(test)
    
    scores = cross_val_score(clf, train, train_labels)
    print 'score.mean(): ', scores.mean()       
    acc = printae(pred, test_labels)    
    ens[2,k] = (acc)
    
    
    print ''
    print '###  BaggingClassifier ###'
    clf = BaggingClassifier( LDA() )
    #clf = BaggingClassifier( svm.SVC(C=8192.0, gamma=8.0, kernel='rbf', probability=True) )
    clf = clf.fit(train, train_labels)
    pred = clf.predict(test)
    
    scores = cross_val_score(clf, train, train_labels)
    print 'score.mean(): ', scores.mean()       
    acc = printae(pred, test_labels)    
    ens[3,k] = (acc)

    ''' 
    print ''
    print '### BaggingRegressor ###'
    clf = BaggingRegressor( LDA() )
    #clf = BaggingClassifier( svm.SVC(C=8192.0, gamma=8.0, kernel='rbf', probability=True) )
    clf = clf.fit(train, train_labels)
    pred = clf.predict(test)
    
    scores = cross_val_score(clf, train, train_labels)
    print 'score.mean(): ', scores.mean()       
    acc = printae(pred, test_labels)    
    '''
    
    '''
    print ''
    print '### AdaBoostClassifier ###'
    #clf = AdaBoostClassifier( LDA() )
    #clf = BaggingClassifier( svm.SVC(C=8192.0, gamma=8.0, kernel='rbf', probability=True) )
    #clf = AdaBoostClassifier( svm.SVC(C=8192.0, gamma=0.03125, kernel='rbf', probability=True) )
    clf = AdaBoostClassifier( KNeighborsClassifier(n_neighbors=5) )
    clf = clf.fit(train, train_labels)
    pred = clf.predict(test)
    
    scores = cross_val_score(clf, train, train_labels)
    print 'score.mean(): ', scores.mean()       
    acc = printae(pred, test_labels)    
    ens[4,k] = (acc)
    '''
    
    k += 1
    print '######################################################'


print '######################################################'


for i in range( len(comb) ):
    print 'comb', i, ':'
    print comb[i]
    print np.mean(comb[i])

for i in range( len(ens) ):
    print 'ens', i, ':'
    print ens[i]
    print np.mean(ens[i])

