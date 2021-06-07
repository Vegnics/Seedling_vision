import cv2
#from Sampling import libseedlingdb as seedb
import numpy as np
import pyrealsense2 as rs
from numpy.fft import fft,fftshift,ifft,ifftshift
from pyleafarea import pyAreaCalc,pyTriangulateAndArea
#from Sampling.libseedlingdb import Sample
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from time import time
import pickle
import sys
import os

sys.path.insert(1, '/home/amaranth/Desktop/Robot_UPAO/Seedling_vision/Sampling')
import libseedlingdb as seedb
from libseedlingdb import Sample

_distance=0.46 #ground distance or max distance
_seedlingheight=0.28 #minimum distance

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


class region():
    def __init__(self):
        self.label = None
        self.contour = None
        self.hole = None

class seedling():
    def __init__(self):
        self.area = None
        self.height = None
        self.enclosingBox = None
        self.peakHeight = None


def click_depth_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        d_image = depth_orig.astype(np.float)
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d_image[y, x])
        #[h,s,v] = depth_rgb_hsv[y,x]
        print("X= {x}, Y={y}, Z={z}".format(x=100 * point[0], y=100 * point[1], z=100 * point[2])) #
    return

def hole_filling(mask,thresh=25):
    mask_ = mask
    ## Remove small regions in mask
    contours, hierar = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assignedSeedlingRegions, labeled = assignLabelContour(mask_, contours)
    labels_to_remove = []
    for region in assignedSeedlingRegions:
        if len(region.contour) < thresh:
            labels_to_remove.append(region.label)

    for label in labels_to_remove:
        mask_ = np.where(labeled == label, 0, mask_).astype(np.uint8)

    mask_inv = cv2.bitwise_not(mask_)
    num_labels, labeled = cv2.connectedComponents(mask_inv)
    label2remove = labeled[0,0]
    mask_inv = np.where(labeled == label2remove, 0, mask_inv)
    filled_mask = mask_ + mask_inv
    return filled_mask

def rearrange_contours(contours):
    new_contours=[]
    for cnt in contours:
        contour = []
        for pnt in cnt:
            print(pnt)
            point = [pnt[1],pnt[0]]        #FOR DEBUGGING
            contour.append(point)
        new_contours.append(contour)
    return new_contours

def contourAsArray(contour):
    rearrangedCnt = []
    for pnt in contour:
        point = [pnt[0][1], pnt[0][0]]
        rearrangedCnt.append(point)
    rearrangedCnt = np.array(rearrangedCnt,dtype=np.int)
    return rearrangedCnt

def rearrange_contours_ocv(contours):
    new_contours=[]
    for cnt in contours:
        contour = []
        for pnt in cnt:
            point = [pnt[0][0],pnt[0][1]]
            contour.append(point)
        new_contours.append(contour)
    return new_contours


def select_external_contours(contours,hierarchy):
    selected = []
    for i in range(len(contours)):
        if hierarchy[0,i,3] == -1: #and len(contours[i])<200:
            selected.append(contours[i])
    return selected

def colorize(depth_image):
    _min=0.30
    _max=0.5
    normalized=255.0*(depth_image-_min)/(_max-_min)
    normalized=np.clip(normalized,0,255).astype(np.uint8)
    colorized=cv2.applyColorMap(normalized,cv2.COLORMAP_JET)
    return colorized

def remove_small_contours(contours,thresh):
    final_contours = []
    for i in range(len(contours)):
        if contours[i].shape[0] > thresh:
            final_contours.append(contours[i])
    return final_contours

def calc_distance(depthimg,contour,hole_position):
    moments = cv2.moments(contour)
    cu = int(moments['m10'] / moments['m00'])
    cv = int(moments['m01'] / moments['m00'])
    cnt_point = rs.rs2_deproject_pixel_to_point(intrinsics, [cu, cv], depthimg[cv, cu])
    return ((100*cnt_point[0]-hole_position[0])**2 + (100*cnt_point[1]-hole_position[1])**2)**0.5


def assignLabelContour(mask,contours):
    num_labels, labeled = cv2.connectedComponents(mask)
    assignedRegions = []
    for cnt in contours:
        SeedlingRegion = region()
        SeedlingRegion.contour = cnt
        for label in range(1,num_labels):
            px = tuple([cnt[0,0,1],cnt[0,0,0]]) # Take the first pixel of the contour
            if labeled[px] == label:
                SeedlingRegion.label = label
                assignedRegions.append(SeedlingRegion)
                break
    return assignedRegions,labeled

def getAreaNHeight(labeled_img,label,depthimg,intrinsics):
    _CAM_CONE_DISTANCE = 45.9792              # I set this value according to the depth images
    indexes = np.where(labeled_img == label)
    indexes = list(zip(indexes[0], indexes[1]))
    p2d = np.array(indexes, dtype=np.int)
    p3d = []
    height_aux = 0
    for idx in indexes:
        pnt = rs.rs2_deproject_pixel_to_point(intrinsics, list((idx[1], idx[0])), depthimg[idx])
        height_aux = height_aux + _CAM_CONE_DISTANCE  - depthimg[idx]*100
        p3d.append([pnt[0] * 100, pnt[1] * 100, pnt[2] * 100])
    p3d = np.array(p3d, dtype=np.float32)
    triangles,area = pyTriangulateAndArea(p2d, p3d,0.42)
    meanHeight = height_aux/len(indexes)
    return area,meanHeight


def getEnclosingBox(regions):
    row_min = 720
    row_max = 0
    col_min = 1280
    col_max = 0
    for region in regions:
        cntArray = contourAsArray(region.contour)
        row_min_aux = np.min(cntArray[:,0])
        row_max_aux = np.max(cntArray[:,0])
        col_min_aux = np.min(cntArray[:,1])
        col_max_aux = np.max(cntArray[:,1])
        if row_min_aux < row_min:
            row_min = row_min_aux
        if row_max_aux > row_max:
            row_max = row_max_aux
        if col_min_aux < col_min:
            col_min = col_min_aux
        if col_max_aux > col_max:
            col_max = col_max_aux
    return [tuple([col_min,row_min]),tuple([col_max,row_max])]


def assign_to_seedling(mask,contours,depthimg,hole_positions,maxdist,parity,intrinsics):
    assignedSeedlingRegions,labeled = assignLabelContour(mask,contours)
    Regions_0 = []
    Regions_1 = []
    Regions_2 = []
    if parity is "even":
        target_holes = [1,3,5]
    elif parity is "odd":
        target_holes = [0, 2, 4]
    else:
        raise TypeError("parity can only be even or odd")

    for region in assignedSeedlingRegions: #Assign a hole to each region
        for idx in target_holes:
            if calc_distance(depthimg,region.contour,hole_positions[idx])<= maxdist:
                region.hole = idx

    for region in assignedSeedlingRegions: # Gather all the regions with the same assigned hole
        if region.hole == target_holes[0]:
            Regions_0.append(region)
        elif region.hole == target_holes[1]:
            Regions_1.append(region)
        elif region.hole == target_holes[2]:
            Regions_2.append(region)


    #CREATE 3 SEEDLING OBJECTS
    Seedling_0 = seedling()
    Seedling_1 = seedling()
    Seedling_2 = seedling()

    ##FIND THE ENCLOSING BOX
    Seedling_0.enclosingBox = getEnclosingBox(Regions_0)
    Seedling_1.enclosingBox = getEnclosingBox(Regions_1)
    Seedling_2.enclosingBox = getEnclosingBox(Regions_2)

    ##AREA AND HEIGHT ESTIMATION FOR EACH SEEDLING
    area = 0
    height=0
    for region in Regions_0:
        area_aux,height_aux = getAreaNHeight(labeled,region.label,depthimg,intrinsics)
        height = height + height_aux
        area = area + area_aux
    Seedling_0.area = area
    if len(Regions_0)>0:
        Seedling_0.height = height/len(Regions_0)
    else:
        Seedling_0.height = height

    area = 0
    height = 0
    for region in Regions_1:
        area_aux, height_aux = getAreaNHeight(labeled, region.label, depthimg, intrinsics)
        height = height + height_aux
        area = area + area_aux
    Seedling_1.area = area
    if len(Regions_1) > 0:
        Seedling_1.height = height/len(Regions_1)
    else:
        Seedling_1.height = height

    area = 0
    height = 0
    for region in Regions_2:
        area_aux, height_aux = getAreaNHeight(labeled, region.label, depthimg, intrinsics)
        height = height + height_aux
        area = area + area_aux
    Seedling_2.area = area
    if len(Regions_2) > 0:
        Seedling_2.height = height/len(Regions_2)
    else:
        Seedling_2.height = height

    return Seedling_0,Seedling_1,Seedling_2

args = sys.argv

if "-debug" in args:
    print("Entering debug mode ...")


##Set the dataset folder
#folder="/home/amaranth/Desktop/Robot_UPAO/seedling_dataset_06_03_2021/"
dataset_date = "06_03_2021"
folder="/home/amaranth/Desktop/Robot_UPAO/seedling_dataset_"+ dataset_date + "/"

#GUI and mouse
cv2.namedWindow("original")
cv2.setMouseCallback("original", click_depth_rgb)

#Set the sample number, save to data and parity
sample_num= 15
seedlingParity = "odd"

save_img = False
save_subimg = False

#Open the sample
DB=seedb.seedling_dataset(folder)
sample=DB.read_sample(sample_num)
depth_orig=sample.depth
depth_rgb=sample.toprgb
depth_rgb_orig = depth_rgb.copy()
mask = np.zeros(depth_rgb.shape[0:2],dtype=np.uint8)
depth_rgb_padded = np.zeros(depth_rgb.shape,dtype=np.uint8)


## OPEN MODELS
file = open("../NewColorSegmentation/MLmodel.pkl", "rb")
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

#### SEGMENTATION USING DEPTH
mask_depth_roi = np.where((depth_roi<0.465)&(depth_roi>0.28),255,0).astype(np.uint8)# pixels between 3cm and 33 cm
preseg_rgb_roi = cv2.bitwise_and(rgb_roi,rgb_roi,mask=mask_depth_roi)

#### SEGMENTATION USING COLOR
preseg_hsv_roi = cv2.cvtColor(preseg_rgb_roi,cv2.COLOR_BGR2HSV) #Convert image to HSV
reshaped_hsv_roi = np.reshape(preseg_hsv_roi,(preseg_hsv_roi.shape[0]*preseg_hsv_roi.shape[1],3))# Reshape image to be used by Kmeans
labeled = segmentation_model.predict(reshaped_hsv_roi)
mask_roi = np.where((labeled==1)|(labeled==3)|(labeled==6),255,0).astype(np.uint8)
mask_roi = np.reshape(mask_roi,(preseg_hsv_roi.shape[0],preseg_hsv_roi.shape[1]))
mask[row_roi:,col_roi[0]:col_roi[1]] = mask_roi

mask = hole_filling(mask,25) # Hole filling

### OBTAIN CONTOURS
contours,hierar = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours = remove_small_contours(contours,60)
cv2.drawContours(depth_rgb, contours, -1, [100, 60, 200], 2)

hole_positions = [[-8.91548333333333,12.2185666666667],[-4.349625,11.0752125],[0.362765555555556,10.9953444444444],[4.84777,11.11838],[9.30483333333333,11.2195888888889],[14.1567625,11.200375]]

## ASSIGN REGIONS AND CONTOURS TO SEEDLING HOLES AND CLASSIFY THEM
S0,S1,S2 = assign_to_seedling(mask,contours,depth_orig,hole_positions,3.8,seedlingParity,intrinsics)
q0 = seedling_classifier.predict([[S0.area,S0.height]])
q1 = seedling_classifier.predict([[S1.area,S1.height]])
q2 = seedling_classifier.predict([[S2.area,S2.height]])

print("S0: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}".format(S0.area,S0.height,q0))
print("S1: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}".format(S1.area,S1.height,q1))
print("S2: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}".format(S2.area,S2.height,q2))




cv2.rectangle(depth_rgb,*S0.enclosingBox,[255,0,0],2)
cv2.rectangle(depth_rgb,*S1.enclosingBox,[255,0,0],2)
cv2.rectangle(depth_rgb,*S2.enclosingBox,[255,0,0],2)


cv2.imshow("original",depth_rgb)
cv2.imshow("orig",depth_rgb_orig)
#cv2.imshow("mask",mask)
#cv2.imshow("mask_depth",mask_depth_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()