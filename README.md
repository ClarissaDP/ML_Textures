
Ana Paula Lemos Rocha

Clarissa Dreischerf Pereira

***************************************
CI171  

Aprendizado de Máquina  

Segundo semestre de 2016  

Trabalho prático - Imagens de textura  
***************************************


**Dados:** /nobackup/prof/lesoliveira/data/ci171


#### Extração:  
> - feature_extraction/lbp.py = local_binary_pattern: http://hanzratech.in/2015/05/30/local-binary-patterns.html  
> - feature_extraction/glcm.py = Grey Level Co-occurrence Matrices: http://scikit-image.org/docs/dev/auto_examples/plot_glcm.html  
> - feature_extraction/histCor.py = Histograma de Cor  
> - feature_extraction/pca.py = http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html  
>
> * Mais = /nobackup/ibm/cdp13/cdp13/nobackup/aprendMaq/tessa  ( https://github.com/ebenolson/tessa ) ?  

#### Classificadores:  
> - classifiers/cross_validation.py =   
>     Decision Tree ( tree.DecisionTreeClassifier() )  
>     Gaussian Naive Bayes ( GaussianNB() )  
>     KNN ( KNeighborsClassifier(n_neighbors=k) )  
>     SVM ( svm.SVC(C=?, gamma=?, kernel='rbf', probability=True) )  
>     Perceptron ( linear_model.Perceptron() )  
>     LDA ( LDA() )  
>     Logistic Regression ( linear_model.LogisticRegression() )   
>       * ovr e multinomial  
> - classifiers/cv_comb_ens.py =   
>     3 combinações  
>     3 ensembles ...  

