#!/usr/bin/python

import cv2, os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

from sklearn.feature_extraction import image


# http://hanzratech.in/2015/05/30/local-binary-patterns.html

# Local Binary Pattern function 
from skimage.feature import local_binary_pattern 
# To calculate a normalized histogram 
from scipy.stats import itemfreq 
from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA



# ************************* Main *************************** #
if ( len(sys.argv) < 2 ):
    print "python pca.py imagesPath"
    exit(-1)

path = sys.argv[1]

robusto = 0
if ( len(sys.argv) > 2 ):
    robusto = 1
        

images = np.array([])
subjects_paths_first = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
subjects_paths_first.sort()
		
#print 'Start reading...'
for s,subjects_paths in enumerate(subjects_paths_first, start=1):
    
    subject_path = [os.path.join(subjects_paths, f) for f in os.listdir(subjects_paths) if f.endswith('.jpg') and os.path.isfile(os.path.join(subjects_paths,f)) ]
    subject_path.sort()
    #print subject_path

    for image_path in subject_path:

        #print ''
	#print '***********************************************************'
        #print 'Agora: ', image_path
        
        img = cv2.imread(image_path, 0)
        #print img.shape 
        h, w = img.shape
        k = h / 8

        pca = PCA(n_components=k)
        pca.fit(img)
        pred = (pca.explained_variance_ratio_)
        
        for val in pred:
            print val, 

        # Cut label
        labelz =  image_path.split('/')
        label_now = labelz[len(labelz)-2]
        print label_now
        
