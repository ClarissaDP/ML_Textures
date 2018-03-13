#!/usr/bin/python

import cv2, os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

from sklearn.feature_extraction import image
from sklearn.preprocessing import normalize



# ************************* Main *************************** #
if ( len(sys.argv) < 2 ):
    print "python lbp.py imagesPath"
    exit(-1)

path = sys.argv[1]

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
             
        '''
        # http://scikit-learn.org/stable/modules/feature_extraction.html
        patches = image.extract_patches_2d(img, (1, 4), max_patches=4)
        print patches.shape
        print patches 
         
        patches = image.extract_patches_2d(img, (2, 2), max_patches=4)
        print patches.shape
        print patches 
        '''
        

        # Cut label
        labelz =  image_path.split('/')
        label_now = labelz[len(labelz)-2]
       
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
    
        '''
    	plt.subplot(111),plt.imshow(lbp,cmap = 'gray')
	plt.title(image_path), plt.xticks([]), plt.yticks([])

	plt.pause(0.5)
        '''

        hist = hist.astype(int)
        #print x.shape, hist.shape
        for i in range( len(hist) ):
            print hist[i][0], 
        print label_now
        print len(hist)

