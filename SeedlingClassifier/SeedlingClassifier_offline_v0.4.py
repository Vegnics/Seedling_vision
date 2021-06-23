import cv2
import numpy as np
import pyrealsense2 as rs
from pyleafarea import pyAreaCalc,pyTriangulateAndArea
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle
import sys
import os
from common_functions import *

from matplotlib import pyplot as plt
sys.path.insert(1, '../')
from Sampling import libseedlingdb as seedb
from Sampling.libseedlingdb import Sample

#set depth camera parameters
distance=0.359
depth_scale=9.999999747378752e-05
intrinsics=rs.intrinsics()
intrinsics.width=1280
intrinsics.height=720
intrinsics.ppx=639.399
intrinsics.ppy=350.551
intrinsics.fx=906.286
intrinsics.fy=905.369
intrinsics.model=rs.distortion.inverse_brown_conrady
intrinsics.coeffs=[0.0,0.0,0.0,0.0,0.0]

#It's possible to click on the depth image in order to know the spatial position of a pixel.


def click_depth_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        d_image = depth_orig.astype(np.float)
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d_image[y, x])
        #[h,s,v] = depth_rgb_hsv[y,x]
        print("X= {x}, Y={y}, Z={z}".format(x=100 * point[0], y=100 * point[1], z=100 * point[2])) #
    return


args = sys.argv

if "-debug" in args:
    print("Entering debug mode ...")


##Set the dataset folder
#folder="/home/amaranth/Desktop/Robot_UPAO/seedling_dataset_06_03_2021/"
dataset_date = "06_03_2021"
folder="/home/amaranth/Desktop/Robot_UPAO/seedling_dataset_"+ dataset_date + "/"

#GUI and mouse
cv2.namedWindow("results")
cv2.setMouseCallback("results", click_depth_rgb)

#Set the sample number, save to data and parity
sample_num= 25
seedlingParity = "odd"

save_img = False
save_subimg = False

#Open the sample
DB=seedb.seedling_dataset(folder)
sample=DB.read_sample(sample_num)
depth_orig=sample.depth #np.load("someImages/IMG_19_13_58.npy")#
depth_rgb=sample.toprgb #cv2.imread("someImages/IMG_19_13_58.jpg",1)#
depth_rgb_orig = depth_rgb.copy()
mask = np.zeros(depth_rgb.shape[0:2],dtype=np.uint8)
mask_cones = np.zeros(depth_rgb.shape[0:2],dtype=np.uint8)
depth_rgb_padded = np.zeros(depth_rgb.shape,dtype=np.uint8)

kernel = highpass_butterworth_kernel(270,840,0.65,1.0,10,2)
reference = cv2.imread("rgb_reference.jpg",0)

## OPEN MODELS
file = open("../Deprecated/MLmodel_v0.3.pkl", "rb")
segmentation_model = pickle.load(file)
file2 = open("Seedling_Classifier_model.pkl", "rb")
seedling_classifier = pickle.load(file2)


###Use the ROI
# Define the ROI
row_roi=450
col_roi=[360,1200]

depth_rgb_padded[row_roi:-1,col_roi[0]:col_roi[1]] = depth_rgb[row_roi:-1,col_roi[0]:col_roi[1]]
depth_roi = depth_orig[row_roi:,col_roi[0]:col_roi[1]]
rgb_roi = depth_rgb_padded[row_roi:,col_roi[0]:col_roi[1]]

rgb_roi = preprocess(rgb_roi,kernel,reference)

#### SEGMENTATION USING DEPTH
mask_depth_roi = np.where((depth_roi<0.47)&(depth_roi>0.28),255,0).astype(np.uint8)# pixels between 3cm and 33 cm
preseg_rgb_roi = cv2.bitwise_and(rgb_roi,rgb_roi,mask=mask_depth_roi)

#### SEGMENTATION USING COLOR
preseg_hsv_roi = cv2.cvtColor(preseg_rgb_roi,cv2.COLOR_BGR2HSV) #Convert image to HSV
reshaped_hsv_roi = np.reshape(preseg_hsv_roi,(preseg_hsv_roi.shape[0]*preseg_hsv_roi.shape[1],3))# Reshape image to be used by Kmeans
labeled = segmentation_model.predict(reshaped_hsv_roi)
#mask_roi = np.where((labeled==1)|(labeled==3)|(labeled==6),255,0).astype(np.uint8)
#mask_roi = np.where((labeled==1)|(labeled==4)|(labeled==6)|(labeled==7)|(labeled==8),255,0).astype(np.uint8) #CHANGED THIS LINE
mask_roi = np.where((labeled==1)|(labeled==4)|(labeled==5)|(labeled==6)|(labeled==7)|(labeled==9)|(labeled==11)|(labeled==13),255,0).astype(np.uint8)
mask_roi = np.reshape(mask_roi,(preseg_hsv_roi.shape[0],preseg_hsv_roi.shape[1]))



#mask_cones_roi = np.where((labeled==2)|(labeled==9)|(labeled==3),255,0).astype(np.uint8) #CHANGED THIS LINE
mask_cones_roi = np.where((labeled==2)|(labeled==3)|(labeled==10),255,0).astype(np.uint8)
mask_cones_roi = np.reshape(mask_cones_roi,(preseg_hsv_roi.shape[0],preseg_hsv_roi.shape[1]))

mask[row_roi:,col_roi[0]:col_roi[1]] = mask_roi
mask = hole_filling(mask,60) # Hole filling

mask_roi = mask[row_roi:,col_roi[0]:col_roi[1]] # <-------------------

seg_rgb_roi = cv2.bitwise_and(rgb_roi,rgb_roi,mask=mask_roi)# <--------------


mask_cones[row_roi:,col_roi[0]:col_roi[1]] = mask_cones_roi

### OBTAIN CONTOURS
contours,hierar = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#contours,hierar = remove_small_contours(contours,hierar,60)

cv2.drawContours(depth_rgb, contours, -1, [100, 60, 200], 2)

hole_positions = [[-8.91548333333333,12.2185666666667],[-4.349625,11.0752125],[0.362765555555556,10.9953444444444],[4.84777,11.11838],[9.30483333333333,11.2195888888889],[14.1567625,11.200375]]

## ASSIGN REGIONS AND CONTOURS TO SEEDLING HOLES AND CLASSIFY THEM
cone_distances = estimate_cones_distances(mask_cones,depth_orig,seedlingParity) # Estimate the distance between camera and cone
print("0 Cone's Distance: {:2.3f} cm".format(cone_distances[0]))
print("1 Cone's Distance: {:2.3f} cm".format(cone_distances[1]))
print("2 Cone's Distance: {:2.3f} cm".format(cone_distances[2]))


S0,S1,S2 = assign_to_seedling2(mask,contours,hierar,depth_orig,hole_positions,3.8,seedlingParity,intrinsics,cone_distances)


q0 = seedling_classifier.predict([[S0.area,S0.height]])
q1 = seedling_classifier.predict([[S1.area,S1.height]])
q2 = seedling_classifier.predict([[S2.area,S2.height]])

print("S0: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}".format(S0.area,S0.height,q0))
print("S1: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}".format(S1.area,S1.height,q1))
print("S2: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}".format(S2.area,S2.height,q2))


cv2.rectangle(depth_rgb,*S0.enclosingBox,[255,0,0],2)
cv2.rectangle(depth_rgb,*S1.enclosingBox,[255,0,0],2)
cv2.rectangle(depth_rgb,*S2.enclosingBox,[255,0,0],2)


#cv2.imshow("original",rgb_roi)
#cv2.imshow("preseg",preseg_rgb_roi)
#cv2.imshow("seg",seg_rgb_roi)
cv2.imshow("mask",mask_roi)
#cv2.imshow("mask_depth",mask_depth_roi)
cv2.imshow("results",depth_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()