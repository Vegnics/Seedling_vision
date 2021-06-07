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
from numpy.fft import fft2,fftshift,ifft2,ifftshift
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt

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
    mask_ = mask.copy()
    ## Remove small regions in mask
    contours, hierar = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assignedRegions, labeled = assignLabelContour(mask_, contours)
    labels_to_remove = []
    for region in assignedRegions:
        if len(region.contour) < thresh:
            labels_to_remove.append(region.label)
    for label in labels_to_remove:
        mask_ = np.where(labeled == label, 0, mask_).astype(np.uint8)
    mask_inv = cv2.bitwise_not(mask_)
    num_labels, labeled = cv2.connectedComponents(mask_inv)
    label2remove = labeled[0,0]
    mask_inv = np.where(labeled == label2remove, 0, mask_inv)
    contours, hierar = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #<<<<<<<<< UNDER DEVELOPMENT
    assignedRegions, labeled = assignLabelContour(mask_inv, contours)
    labels_to_remove = [] #labels_to_add
    for region in assignedRegions:
        if len(region.contour) > thresh: #if len(region.contour) > thresh:
            labels_to_remove.append(region.label)
    for label in labels_to_remove:
        mask_inv = np.where(labeled == label,0,mask_inv).astype(np.uint8)
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

def remove_small_contours(contours,hierarchy,thresh):
    final_contours = []
    final_hierarchy = []
    final_hierarchy_= []
    for i in range(len(contours)):
        if contours[i].shape[0] > thresh:
            final_contours.append(contours[i])
            final_hierarchy_.append(list(hierarchy[0][i]))
    final_hierarchy.append(final_hierarchy_)
    final_hierarchy = np.array(final_hierarchy,dtype=np.int32)
    return final_contours,final_hierarchy

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

def getAreaNHeight(labeled_img,label,depthimg,intrinsics,cone_distance):
    _CAM_CONE_DISTANCE = cone_distance #45.9792              # I set this value according to the depth images
    indexes = np.where(labeled_img == label)
    indexes = list(zip(indexes[0], indexes[1]))
    p2d = np.array(indexes, dtype=np.int)
    p3d = []
    height_aux = 0
    valid_height_values = 0
    for idx in indexes:
        pnt = rs.rs2_deproject_pixel_to_point(intrinsics, list((idx[1], idx[0])), depthimg[idx])
        if (100*pnt[2])<_CAM_CONE_DISTANCE:
            height_aux = height_aux + _CAM_CONE_DISTANCE  - depthimg[idx]*100
            valid_height_values += 1
        p3d.append([pnt[0] * 100, pnt[1] * 100, pnt[2] * 100])
    p3d = np.array(p3d, dtype=np.float32)
    triangles,area = pyTriangulateAndArea(p2d, p3d,0.42)
    if valid_height_values > 0:
        meanHeight = height_aux/valid_height_values
    else:
        meanHeight = 0.0
    return area,meanHeight # I added triangles on 31/05


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


def assign_to_seedling(mask,contours,hierarchy,depthimg,hole_positions,maxdist,parity,intrinsics):
    contours = select_external_contours(contours,hierarchy)
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
        area_aux,height_aux = getAreaNHeight(labeled,region.label,depthimg,intrinsics,1)
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
        area_aux, height_aux = getAreaNHeight(labeled, region.label, depthimg, intrinsics,1)
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
        area_aux, height_aux = getAreaNHeight(labeled, region.label, depthimg, intrinsics,1)
        height = height + height_aux
        area = area + area_aux
    Seedling_2.area = area
    if len(Regions_2) > 0:
        Seedling_2.height = height/len(Regions_2)
    else:
        Seedling_2.height = height

    return Seedling_0,Seedling_1,Seedling_2

def assign_to_seedling2(mask,contours,hierarchy,depthimg,hole_positions,maxdist,parity,intrinsics,cone_distances):
    contours = select_external_contours(contours, hierarchy)
    assignedSeedlingRegions,labeled = assignLabelContour(mask,contours)
    Regions_0 = []
    Regions_1 = []
    Regions_2 = []
    if parity is "odd":
        target_holes = [1,3,5]
    elif parity is "even":
        target_holes =  [0, 2, 4]
    else:
        raise TypeError("parity can only be even or odd")

    for region in assignedSeedlingRegions: #Assign a hole to each region
        hole_distances=[]
        for idx in target_holes:
            hole_distances.append(calc_distance(depthimg,region.contour,hole_positions[idx]))#<= maxdist:
        if hole_distances[0]<= hole_distances[1] and hole_distances[0]<=hole_distances[2]:
            region.hole = 0
        elif hole_distances[1]<= hole_distances[0] and hole_distances[1]<=hole_distances[2]:
            region.hole = 1
        elif hole_distances[2] <= hole_distances[1] and hole_distances[2] <= hole_distances[0]:
            region.hole = 2

    for region in assignedSeedlingRegions: # Gather all the regions with the same assigned hole
        if region.hole == 0:
            Regions_0.append(region)
        elif region.hole == 1:
            Regions_1.append(region)
        elif region.hole == 2:
            Regions_2.append(region)


    #CREATE 3 SEEDLING OBJECTS
    Seedling_0 = seedling()
    Seedling_1 = seedling()
    Seedling_2 = seedling()

    ##FIND THE ENCLOSING BOX
    Seedling_2.enclosingBox = getEnclosingBox(Regions_0) #Seedling_0.enclosingBox = getEnclosingBox(Regions_0)
    Seedling_1.enclosingBox = getEnclosingBox(Regions_1)
    Seedling_0.enclosingBox = getEnclosingBox(Regions_2)

    ##AREA AND HEIGHT ESTIMATION FOR EACH SEEDLING
    area = 0
    height=0
    for region in Regions_0:
        area_aux,height_aux = getAreaNHeight(labeled,region.label,depthimg,intrinsics,cone_distances[0])
        height = height + height_aux
        area = area + area_aux
    Seedling_2.area = area #Seedling_0.area = area
    if len(Regions_0)>0:
        Seedling_2.height = height/len(Regions_0)
    else:
        Seedling_2.height = height

    area = 0
    height = 0
    for region in Regions_1:
        area_aux, height_aux = getAreaNHeight(labeled, region.label, depthimg, intrinsics,cone_distances[1])
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
        area_aux, height_aux = getAreaNHeight(labeled, region.label, depthimg, intrinsics,cone_distances[2])
        height = height + height_aux
        area = area + area_aux
    Seedling_0.area = area #Seedling_2.area = area
    if len(Regions_2) > 0:
        Seedling_0.height = height/len(Regions_2)
    else:
        Seedling_0.height = height

    return Seedling_0,Seedling_1,Seedling_2

def estimate_cones_distances(mask,depthimg,parity):
    ## The ranges specified below depends on the current camera position/orientation. If the camera were moved
    ## it would be neccessary to changes this ranges values according to the cones' positions.
    if parity is "odd":
        cones_mask_roi=[mask[515:630,850:960],mask[515:630,670:780],mask[515:630,500:610]]
        cones_depth_roi =[depthimg[515:630,850:960],depthimg[515:630,670:780],depthimg[515:630,500:610]]

    elif parity is "even":
        cones_mask_roi=[mask[515:630, 770:880], mask[515:630, 580:690], mask[515:630, 415:525]]
        cones_depth_roi=[depthimg[515:630, 770:880], depthimg[515:630, 580:690], depthimg[515:630, 415:525]]

    cones_distance=[]
    for cone_idx in range(3):
        roi = cones_mask_roi[cone_idx]
        cone_depthimg = 100*cones_depth_roi[cone_idx]
        cone_depthimg_aux = np.where(roi==255,cone_depthimg,0)
        cone_depthimg_aux = np.where((47>cone_depthimg_aux) & (cone_depthimg_aux>44.5),cone_depthimg_aux,0)
        masked_values = cone_depthimg_aux.astype(np.bool).astype(np.float)
        num_of_valids= masked_values.sum()
        depth_sum = cone_depthimg_aux.sum()
        if num_of_valids > 0:
            distance = depth_sum / num_of_valids
            cones_distance.append(distance)
        else:
            cones_distance.append(45.5)
    return cones_distance


def highpass_butterworth_kernel(size0,size1,sl,sh,rc,n):
    kernel = np.zeros((size0,size1))
    for i in range(size0):
        for j in range(size1):
            kernel[i,j] = sh + (sl-sh)/(1+2.415*(((((i-int(size0/2))**2 + (j-int(size1/2))**2)**0.5)/rc))**(2*n))
    return kernel

def homomorph_filter_N1(src,kernel):
    src = src.astype(np.float32)
    Ln_I = np.log(src + 1)
    I_fft = fft2(Ln_I)
    I_fft = fftshift(I_fft)
    #kernel = highpass_gaussian_kernel(I_fft.shape[0], I_fft.shape[1], sigma)
    I_filt_fft = I_fft * kernel
    I_filt_fft_uns = ifftshift(I_filt_fft)
    I_filtered = np.real(ifft2(I_filt_fft_uns))
    I_filtered = np.exp(I_filtered) - 1
    return I_filtered,np.min(I_filtered),np.max(I_filtered)

def homomorph_filter_N3(src,kernel):
    outimg = np.zeros(src.shape)
    B, G, R = cv2.split(src)
    nB,minB,maxB = homomorph_filter_N1(B, kernel)
    nG,minG,maxG = homomorph_filter_N1(G, kernel)
    nR,minR,maxR = homomorph_filter_N1(R, kernel)
    outimg[:, :, 0] = nB
    outimg[:, :, 1] = nG
    outimg[:, :, 2] = nR
    return outimg

def preprocess(src,kernel,reference):
    src_hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    val = src_hsv[:,:,2]
    I_filtered = homomorph_filter_N3(src,kernel)
    I_filtered = np.clip(I_filtered, 0.0, 255.0)
    I_filtered = np.uint8(I_filtered)
    img_hsv = cv2.cvtColor(I_filtered, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = match_histograms(img_hsv[:, :, 2],reference)
    I_filtered = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return I_filtered

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
sample_num= 30
seedlingParity = "even"

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
reference = cv2.imread("/home/amaranth/Desktop/Robot_UPAO/Seedling_vision/segmentation/clustering_with_preproc/imgsNmasks/rgb_reference.jpg",0)

## OPEN MODELS
file = open("../segmentation/clustering_with_preproc/MLmodel_v0.3.pkl", "rb")
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


cv2.imshow("original",rgb_roi)
#cv2.imshow("preseg",preseg_rgb_roi)
#cv2.imshow("seg",seg_rgb_roi)
#cv2.imshow("mask",mask_roi)
#cv2.imshow("mask_depth",mask_depth_roi)
cv2.imshow("results",depth_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()