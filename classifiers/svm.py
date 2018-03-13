#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing



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
        


def main(tr, te, opcao, mi, ma):

        # loads data
        print "Loading data..."

        #train, train_labels = load_svmlight_file(tr)
        #test, test_labels = load_svmlight_file(te)


        # Open and access the file
        tr_f = tr 
        te_f = te

        if (not mi):
            mi = 120
        if (not ma):
            ma = 121

        print mi, ma

        print 'Starting read from Train file...'
        temp = np.loadtxt(tr_f, dtype=float, skiprows=1)
        print temp.shape
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

        test_labels = test_labels.ravel()
        train_labels = train_labels.ravel()
    
        print test_labels
        print test_labels.shape

        # GridSearch retorna o melhor modelo encontrado na busca
        if (opcao):
            #print 'ENTROU'
            best = GridSearch(train, train_labels)
        else:
            #print 'PASSOU'
            best = svm.SVC(C=8192.0, gamma=8.0,kernel='rbf', probability=True)

        # Treina usando o melhor modelo
        best.fit(train, train_labels)

        # resultado do treinamento
        print best.score(test, test_labels)

        # predicao do classificador
        pred = best.predict(test)

        # cria a matriz de confusao
        cm = confusion_matrix(test_labels, pred)
        print cm

        # probabilidades
        #probs = best.predict_proba(test)
        #print probs


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Use: svm.py <tr> <ts>")
    
    if len(sys.argv) == 4:
        print '3'
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), 0, 0)
    
    if len(sys.argv) > 4:
        print '4'
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) )


    main(sys.argv[1], sys.argv[2], 0, 0, 0)
