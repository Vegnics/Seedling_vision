import cv2
import numpy as np
import pyrealsense2 as rs
from pyleafarea import pyAreaCalc,pyTriangulateAndArea
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from time import time,sleep,localtime
import pickle
import sys
from numpy.fft import fft2,fftshift,ifft2,ifftshift
from skimage.exposure import match_histograms

sys.path.insert(1,"../")
from modbus_mqtt.libseedlingmodbus import SeedlingModbusClient
from modbus_mqtt import libseedlingmodbus as lsmodb

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


def hole_filling(mask,thresh=25):
    mask_ = mask
    ## Remove small regions in mask
    if cv2.__version__ >= "4.0":
        contours, hierar = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _,contours, hierar = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

def contourAsArray(contour):
    rearrangedCnt = []
    for pnt in contour:
        point = [pnt[0][1], pnt[0][0]]
        rearrangedCnt.append(point)
    rearrangedCnt = np.array(rearrangedCnt,dtype=np.int)
    return rearrangedCnt

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

def calc_distance(depthimg,contour,hole_position,intrinsics):
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


def select_external_contours(contours,hierarchy):
    selected = []
    for i in range(len(contours)):
        if hierarchy[0,i,3] == -1: #and len(contours[i])<200:
            selected.append(contours[i])
    return selected

def assign_to_seedling2(mask,contours,hierarchy,depthimg,hole_positions,maxdist,parity,intrinsics,cone_distances):
    contours = select_external_contours(contours, hierarchy)
    assignedSeedlingRegions,labeled = assignLabelContour(mask,contours)
    Regions_0 = []
    Regions_1 = []
    Regions_2 = []
    if parity is "odd":
        target_holes = [1, 3, 5]# I've changed the values from [1,3,5] on 31/05/2021
    elif parity is "even":
        target_holes = [0, 2, 4] # I've changed the values from [0, 2, 4] on 31/05/2021
    else:
        raise TypeError("parity can only be even or odd")

    for region in assignedSeedlingRegions: #Assign a hole to each region
        hole_distances=[]
        for idx in target_holes:
            hole_distances.append(calc_distance(depthimg,region.contour,hole_positions[idx],intrinsics))#<= maxdist:
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
    Seedling_2.area = area
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
    Seedling_0.area = area
    if len(Regions_2) > 0:
        Seedling_0.height = height/len(Regions_2)
    else:
        Seedling_0.height = height

    return Seedling_0,Seedling_1,Seedling_2

def estimate_cones_distances(mask,depthimg,parity):
    ## The ranges specified below depends on the current camera position/orientation. If the camera were moved
    ## it would be neccessary to changes this ranges values according to the cones' positions.
    ROI_OK = False
    if parity is "odd":
        cones_mask_roi=[mask[515:630,850:960],mask[515:630,670:780],mask[515:630,500:610]]
        cones_depth_roi =[depthimg[515:630,850:960],depthimg[515:630,670:780],depthimg[515:630,500:610]]

    elif parity is "even":
        cones_mask_roi=[mask[515:630, 770:880], mask[515:630, 580:690], mask[515:630, 415:525]]
        cones_depth_roi=[depthimg[515:630, 770:880], depthimg[515:630, 580:690], depthimg[515:630, 415:525]]
    else:
        raise Exception("parity can only be odd or even")

    cones_distance=[]
    for cone_idx in range(3):
        roi = cones_mask_roi[cone_idx]
        cone_depthimg = 100*cones_depth_roi[cone_idx]
        cone_depthimg_aux = np.where(roi==255,cone_depthimg,0)
        cone_depthimg_aux = np.where((48.6>cone_depthimg_aux) & (cone_depthimg_aux>45.66),cone_depthimg_aux,0)
        masked_values = cone_depthimg_aux.astype(np.bool).astype(np.float)
        num_of_valids= masked_values.sum()
        depth_sum = cone_depthimg_aux.sum()
        if num_of_valids > 0:
            distance = depth_sum / num_of_valids
            cones_distance.append(distance)
        else:
            cones_distance.append(46.16)
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
    I_filtered = homomorph_filter_N3(src,kernel)
    I_filtered = np.clip(I_filtered, 0.0, 255.0)
    I_filtered = np.uint8(I_filtered)
    img_hsv = cv2.cvtColor(I_filtered, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = match_histograms(img_hsv[:, :, 2],reference)
    I_filtered = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return I_filtered

class seedlingClassifier():
    def __init__(self,segmentationModel,seedlingClassifierModel,intrinsics):
        self.modbusClient = None
        self.__rspipeline = None
        self.__rsconfig = None
        self.__align = rs.align(rs.stream.color)
        self.modbusConnectedFlag = False
        self.cameraInitializedFlag = False
        self.depthImg = None
        self.rgbImg = None
        self.cvstatus = 0
        self.initial_discard_frames = 30
        self.discard_frames = 5
        self.segmentationModel = segmentationModel
        self.seedlingClassifierModel = seedlingClassifierModel
        self.row_roi = 450
        self.col_roi = [360, 1200]
        self.hole_positions = [[-8.91548333333333,12.2185666666667],[-4.349625,11.0752125],[0.362765555555556,10.9953444444444],[4.84777,11.11838],[9.30483333333333,11.2195888888889],[14.1567625,11.200375]]
        self.intrinsics = intrinsics
        self.depth_scale = 9.999999747378752e-05
        self.kernel = highpass_butterworth_kernel(270,840,0.65,1.0,10,2)
        self.reference = cv2.imread("rgb_reference.jpg",0)
        if self.reference is None:
            raise Exception("RGB reference image not found")

    def modbusConnect(self,serverIp,serverPort): #### I need to know if I'm connected to the server.
        self.modbusClient = SeedlingModbusClient(serverIp,serverPort)
        self.modbusConnectedFlag = self.modbusClient.connectToServer()
        if self.modbusConnectedFlag == True:
            print("Connection to server {}:{} -> successful".format(serverIp, serverPort))
        else:
            self.modbusConnectedFlag = False
            print("Server {}:{} not found".format(serverIp,serverPort))

    def writeSeedlingsQuality(self,q0,q1,q2):
        if self.modbusConnectedFlag is True:
            try:
                self.modbusClient.writeSeedling1Quality(q0)
                self.modbusClient.writeSeedling2Quality(q1)
                self.modbusClient.writeSeedling3Quality(q2)
            except:
                print("Cannot send seedlings quality")
        else:
            print("Cannot send seedlings quality")

    def cameraInitialize(self):
        self.__rspipeline = rs.pipeline()
        self.__rsconfig = rs.config()
        self.__rsconfig.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.__rsconfig.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        #First start
        self.__rspipeline_profile = self.__rspipeline.start(self.__rsconfig)
        self.__rsdepth_sensor = self.__rspipeline_profile.get_device().first_depth_sensor()
        self.__rsdepth_sensor.set_option(rs.option.emitter_enabled, 1)
        self.__rsdepth_sensor.set_option(rs.option.laser_power, 250)
        self.__rsdepth_sensor.set_option(rs.option.depth_units, 0.0001)  # changed 0.0001
        self.__temp_filter = rs.temporal_filter()
        self.__temp_filter.set_option(rs.option.filter_smooth_alpha, 0.8)
        self.__temp_filter.set_option(rs.option.filter_smooth_delta, 10)
        self.__temp_filter.set_option(rs.option.holes_fill, 1.0)
        self.__spatial_filter = rs.spatial_filter()
        self.__spatial_filter.set_option(rs.option.holes_fill, 3)
        sleep(0.06)
        self.__rspipeline.stop()
        sleep(0.06)
        # Second start
        self.__rspipeline_profile = self.__rspipeline.start(self.__rsconfig)
        self.__rsdepth_sensor = self.__rspipeline_profile.get_device().first_depth_sensor()
        self.__rsdepth_sensor.set_option(rs.option.emitter_enabled, 1)
        self.__rsdepth_sensor.set_option(rs.option.laser_power, 250)
        self.__rsdepth_sensor.set_option(rs.option.depth_units, 0.0001)  # changed 0.0001
        self.__temp_filter = rs.temporal_filter()
        self.__temp_filter.set_option(rs.option.filter_smooth_alpha, 0.8)
        self.__temp_filter.set_option(rs.option.filter_smooth_delta, 10)
        self.__temp_filter.set_option(rs.option.holes_fill, 1.0)
        self.__spatial_filter = rs.spatial_filter()
        self.__spatial_filter.set_option(rs.option.holes_fill, 3)
        frames = self.__rspipeline.wait_for_frames(timeout_ms=2000)
        aligned_frames = self.__align.process(frames)  # NEW
        for i in range(self.initial_discard_frames):
            depth_frame = aligned_frames.get_depth_frame()  # From frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            depth_frame = self.__spatial_filter.process(depth_frame)
            depth_frame = self.__temp_filter.process(depth_frame)
        self.cameraInitializedFlag = True
    def cameraRestart(self):
        if self.__rspipeline is not None:
            try:
                self.cameraInitialize()
            except Exception as e:
                print("Camera not found: {}".format(e))
    def getImages(self):
        if self.cameraInitializedFlag is False:
            try:
                self.cameraInitialize()
            except Exception as e:
                print("Camera not found: {}".format(e))
        try:
            self.intrinsics = self.__rspipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            frames = self.__rspipeline.wait_for_frames(timeout_ms=2000)
            aligned_frames = self.__align.process(frames)  # NEW
            for i in range(self.discard_frames):
                depth_frame = aligned_frames.get_depth_frame()  # From frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                depth_frame = self.__spatial_filter.process(depth_frame)
                depth_frame = self.__temp_filter.process(depth_frame)
            #    color_frame = frames.get_color_frame()
            # depth_frame = aligned_frames.get_depth_frame() #From frames.get_depth_frame()
            depth_frame = self.__temp_filter.process(depth_frame)
            color_frame = aligned_frames.get_color_frame()  # From frames.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            self.depthImg = self.depth_scale*depth_image
            self.rgbImg = color_image
            return True
        except Exception as e:
            print("Problem while getting images: {}".format(e))
            return False

    def correctZValues(self,conedistances):
        z1correction = (conedistances[0] - 46.16) * 10
        z2correction = (conedistances[1] - 46.16) * 10
        z3correction = (conedistances[2] - 46.16) * 10
        self.modbusClient.writeZcorrection(z1correction,z2correction,z3correction)

    def processSeedlings(self,seedlingParity):
        if self.cameraInitializedFlag is True:
            __processing_start = time()
            if self.getImages() is True:
                rgbGUI = self.rgbImg.copy()
                rgb_padded = np.zeros(self.rgbImg.shape, dtype=np.uint8)
                rgb_padded[self.row_roi:-1, self.col_roi[0]:self.col_roi[1]] = self.rgbImg[self.row_roi:-1, self.col_roi[0]:self.col_roi[1]]
                mask = np.zeros(self.rgbImg.shape[0:2], dtype=np.uint8)
                mask_cones = np.zeros(self.rgbImg.shape[0:2], dtype=np.uint8)

                ##Get the ROI
                depth_roi = self.depthImg[self.row_roi:, self.col_roi[0]:self.col_roi[1]]
                rgb_roi = rgb_padded[self.row_roi:, self.col_roi[0]:self.col_roi[1]]
                rgb_roi = preprocess(rgb_roi,self.kernel,self.reference) ##ADDED THIS LINE

                ##Segmentation using depth
                mask_depth_roi = np.where((depth_roi < 0.473) & (depth_roi > 0.28), 255, 0).astype(np.uint8)  # pixels between 3cm and 33 cm
                preseg_rgb_roi = cv2.bitwise_and(rgb_roi, rgb_roi, mask=mask_depth_roi)

                ##Segmentation using color
                #Segmentation of Seedlings
                preseg_hsv_roi = cv2.cvtColor(preseg_rgb_roi, cv2.COLOR_BGR2HSV)  # Convert image to HSV
                reshaped_hsv_roi = np.reshape(preseg_hsv_roi, (preseg_hsv_roi.shape[0] * preseg_hsv_roi.shape[1], 3))  # Reshape image to be used by Kmeans
                labeled = self.segmentationModel.predict(reshaped_hsv_roi)
                #mask_roi = np.where((labeled == 1) | (labeled == 3) | (labeled == 6), 255, 0).astype(np.uint8)
                #mask_roi = np.where((labeled == 1) | (labeled == 4) | (labeled == 6) | (labeled == 7) | (labeled == 8),255, 0).astype(np.uint8)
                mask_roi = np.where((labeled==1)|(labeled==4)|(labeled==5)|(labeled==6)|(labeled==7)|(labeled==9)|(labeled==11)|(labeled==13),255,0).astype(np.uint8) # <- changed in 02/06
                mask_roi = np.reshape(mask_roi, (preseg_hsv_roi.shape[0], preseg_hsv_roi.shape[1]))
                mask[self.row_roi:, self.col_roi[0]:self.col_roi[1]] = mask_roi
                mask = hole_filling(mask, 25)  # Hole filling

                #Segmentation of Cones
                #mask_cones_roi = np.where((labeled == 2) | (labeled == 9) | (labeled == 3), 255, 0).astype(np.uint8)
                mask_cones_roi = np.where((labeled==2)|(labeled==3)|(labeled==10),255,0).astype(np.uint8) # <- changed in 02/06
                mask_cones_roi = np.reshape(mask_cones_roi, (preseg_hsv_roi.shape[0], preseg_hsv_roi.shape[1]))
                mask_cones[self.row_roi:, self.col_roi[0]:self.col_roi[1]] = mask_cones_roi

                #Obtain camera-cones distances
                cone_distances = estimate_cones_distances(mask_cones, self.depthImg, seedlingParity)

                ##Obtain contours
                if cv2.__version__ >= "4.0":
                    contours, hierar = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                else:
                    _,contours, hierar = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contours,hierar = remove_small_contours(contours,hierar, 45)
                cv2.drawContours(rgbGUI, contours, -1, [100, 60, 200], 2)

                ## ASSIGN REGIONS AND CONTOURS TO SEEDLING HOLES AND CLASSIFY THEM
                S0, S1, S2 = assign_to_seedling2(mask, contours,hierar, self.depthImg, self.hole_positions, 6.9,seedlingParity,self.intrinsics,cone_distances)
                q0 = self.seedlingClassifierModel.predict([[S0.area, S0.height]])
                q1 = self.seedlingClassifierModel.predict([[S1.area, S1.height]])
                q2 = self.seedlingClassifierModel.predict([[S2.area, S2.height]])
                cv2.rectangle(rgbGUI, *S0.enclosingBox, [255, 0, 0], 2)
                cv2.rectangle(rgbGUI, *S1.enclosingBox, [255, 0, 0], 2)
                cv2.rectangle(rgbGUI, *S2.enclosingBox, [255, 0, 0], 2)
                print("Processing Time: {} seconds".format(time()-__processing_start))
                print("S0: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}, Cone distance = {}".format(S0.area, S0.height,q0,cone_distances[0]))
                print("S1: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}, Cone distance = {}".format(S1.area, S1.height,q1,cone_distances[1]))
                print("S2: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}, Cone distance = {}".format(S2.area, S2.height,q2,cone_distances[2]))
                if self.modbusConnectedFlag == True:
                    self.writeSeedlingsQuality(int(q0[0]),int(q1[0]),int(q2[0]))
                    self.correctZValues(cone_distances)
                    self.modbusClient.cvFinishProcessing()
                    print("Results sent to server \n")
                return rgbGUI
            else:
                self.rgbImg = None
                self.depthImg = None
                print("Couldn't get the images")
                return None
        else:
            print("Initialize the camera first")
            return None