#!/usr/bin/python

import cv2, os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

from facerec.lbp import LPQ, ExtendedLBP


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
    
    subject_path = [os.path.join(subjects_paths, f) for f in os.listdir(subjects_paths) if f.endswith('.jpg') and os.path.isfile(os.path.join(subjects_paths,f))]
    subject_path.sort()
    #print subject_path

    for image_path in subject_path:

        #print ''
	#print '***********************************************************'
        #print 'Agora: ', image_path
        
        img = cv2.imread(image_path, 0)

        #:print_line(asm[0])

        labelz =  image_path.split('/')
        label_now = labelz[len(labelz)-2]
        print label_now

