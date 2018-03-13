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
from sklearn.preprocessing import RobustScaler



# ************************* Main *************************** #
if ( len(sys.argv) < 2 ):
    print "python lbp.py imagesPath"
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
             
        '''
        # http://scikit-learn.org/stable/modules/feature_extraction.html
        patches = image.extract_patches_2d(img, (1, 4), max_patches=4)
        print patches.shape
        print patches 
         
        patches = image.extract_patches_2d(img, (2, 2), max_patches=4)
        print patches.shape
        print patches 
        '''
        
        # List for storing the LBP Histograms, address of images and the corresponding label 
        X_test = [] 
        X_name = [] 
        y_test = [] 
        
        # For each image in the training set calculate the LBP histogram 
        # and update X_test, X_name and y_test 8 for train_image in train_images: 
        
        # Convert to grayscale as LBP works on grayscale image 
        #im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        im_gray = img
        radius = 9
        # Number of points to be considered as neighbourers 
        no_points =  8 * radius 
        # Uniform LBP is used 
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform') 
        print lbp.shape
        # Calculate the histogram 
        x = itemfreq(lbp.ravel()) 
        #(hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, no_points + 3), range=(0, no_points + 2))
        # Normalize the histogram 
        #hist = x[:, 1]/sum(x[:, 1]) 
        #eps = 1e-7
        #hist = hist.astype("float")
        #hist /= (hist.sum() + eps)
        
        if (robusto):
            hue = RobustScaler()
            hist = hue.fit_transform(x)
            hist = hist.ravel()
        else:
            # Calculate the histogram 
            #x = itemfreq(lbp.ravel()) 
            # Normalize the histogram 
            hist = x[:, 1]/sum(x[:, 1]) 


        # Cut label
        labelz =  image_path.split('/')
        label_now = labelz[len(labelz)-2]
        #print label_now
        
        # Append image path in X_name
        #X_name.append( image_path ) 
        # Append histogram to X_name 
        #X_test.append( hist ) 
        # Append class label in y_test 
        #y_test.append(train_dic[os.path.split(train_image)[1]]) 
        #y_test.append( label_now ) 
        
    
        '''
    	plt.subplot(111),plt.imshow(lbp,cmap = 'gray')
	plt.title(image_path), plt.xticks([]), plt.yticks([])

	plt.pause(0.5)
        '''

        #print x.shape, hist.shape
        for i in range( len(hist) ):
            print hist[i], 
        print label_now

        # Display the training images 
        '''
        nrows = 512 
        ncols = 512
        fig, axes = plt.subplots(nrows,ncols) 
        for row in range(nrows):
            print "hue"
            for col in range(ncols): 
                axes[row][col].imshow(cv2.cvtColor(cv2.imread(X_name[row*ncols+col]), cv2.COLOR_BGR2RGB)) 
                axes[row][col].axis('off')
                axes[row][col].set_title("{}".format(os.path.split(X_name[row*ncols+col])[1]))
        '''
        


