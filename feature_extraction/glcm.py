#!/usr/bin/python

import cv2, os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

from skimage.feature import greycomatrix, greycoprops


SIZE = 8
p_SIZE = np.pi/SIZE




def print_line( array ):
    for value in (array):
        print value,



# ************************* Main *************************** #
if ( len(sys.argv) < 2 ):
    print "python glcm.py imagesPath"
    exit(-1)

path = sys.argv[1]


images = np.array([])
subjects_paths_first = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
subjects_paths_first.sort()
		
#print 'Start reading...'
for s,subjects_paths in enumerate(subjects_paths_first, start=1):
    

    # CI171TrainVal
    #subject_path = [os.path.join(subjects_paths, f) for f in os.listdir(subjects_paths) if f.endswith('.jpg') and os.path.isfile(os.path.join(subjects_paths,f))]
    
    # Softwood
    subject_path = [os.path.join(subjects_paths, f) for f in os.listdir(subjects_paths) if f.endswith('.png') and os.path.isfile(os.path.join(subjects_paths,f))]
    
    subject_path.sort()
    #print subject_path

    for image_path in subject_path:

        #print ''
	#print '***********************************************************'
        #print 'Agora: ', image_path
        
        img = cv2.imread(image_path, 0)

        '''
        a_loc = np.array([])
        len_img = len(img)
        for i in range(0, len_img, SIZE):
            for j in range(0, len_img, SIZE):
            
                a_loc = np.append(a_loc, img[i:(i+SIZE), j:(j+SIZE)])
        '''

        #for patch in (a_loc):
        #glcm = greycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
        glcm = greycomatrix(img, [1, 5, 10, 15, 20], [0, p_SIZE, 2*p_SIZE, 3*p_SIZE, 4*p_SIZE, 5*p_SIZE, 6*p_SIZE, 7*p_SIZE], 256, symmetric=True, normed=True)
        #diss = (greycoprops(glcm, 'dissimilarity'))
        corr = (greycoprops(glcm, 'correlation'))
        homo = (greycoprops(glcm, 'homogeneity'))
        ener = (greycoprops(glcm, 'energy'))
        #cont = (greycoprops(glcm, 'contrast'))
        #asm = (greycoprops(glcm, 'ASM'))
        #print glcm.shape, diss, corr
       
        #print glcm
        '''
        count = 0
        count_all = np.array([])
        #print len(glcm)
        for k in range( len(glcm) ):
            soma = 0
            for l in range( len(glcm[k]) ):
                soma += np.sum(glcm[k,l])
            
            if soma != 0:
                #print soma, k, l
                count += 1
            count_all = np.append(count_all, soma) 
            #count += 1

        #print 'Total = ', count
        #sys.exit(0)
        
        #for valor in (count_all):
        #    print valor,
        print diss, corr, homo, ener, cont, asm,
        '''

        for i in range( len(corr) ):
            #print_line(diss[i])
            print_line(corr[i])
            print_line(homo[i])
            print_line(ener[i])
            #print_line(cont[i])
            #print_line(asm[i])

        # CI171TrainVal
        #labelz =  image_path.split('/')
        #label_now = labelz[len(labelz)-2]
        #print label_now

        # Softwood
        labelz =  image_path.split('/')
        label_now = labelz[len(labelz)-2]
        label_now =  label_now[:3]
        print label_now
        

