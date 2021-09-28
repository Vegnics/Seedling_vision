import cv2
import numpy as np
import pyrealsense2 as rs
from pyleafarea import pyAreaCalc,pyTriangulateAndArea
from sklearn.cluster import KMeans
from time import time,sleep,localtime
import pickle
import sys
from numpy.fft import fft2,fftshift,ifft2,ifftshift
from skimage.exposure import match_histograms
from skimage.segmentation import slic

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


def colorizeDepth(depth,min,max):
    normalized = 255.0 * (depth - min) / (max - min)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return colorized


def removeSmallRegions(mask,RegionThresh=100):
    mask_ = mask.copy()
    ## Remove small regions in mask
    if cv2.__version__ >= "4.0":
        contours, hierar = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _,contours, hierar = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assignedSeedlingRegions, labeled = assignLabelContour(mask_, contours)
    labels_to_remove = []
    for region in assignedSeedlingRegions:
        #M = cv2.moments(region.contour) #Added 19/06/2021
        if cv2.contourArea(region.contour) < RegionThresh:
            labels_to_remove.append(region.label)
    for label in labels_to_remove:
        mask_ = np.where(labeled == label, 0, mask_).astype(np.uint8)
    return mask_

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
    points = []
    if len(contour)>40:
        for idx in range(0,len(contour),10):
            u = contour[idx,0,0]
            v = contour[idx,0,1]
            if 0.28<depthimg[v,u]<0.60:
                pnt = rs.rs2_deproject_pixel_to_point(intrinsics, [u,v], depthimg[v,u])
                points.append(pnt[0:2])
    else:
        for idx in range(len(contour)):
            u = contour[idx,0,0]
            v = contour[idx,0,1]
            if 0.28<depthimg[v,u]<0.60:
                pnt = rs.rs2_deproject_pixel_to_point(intrinsics, [u,v], depthimg[v,u])
                points.append(pnt[0:2])
    mean_point = np.mean(points,0)
    #cu = int(moments['m10'] / moments['m00'])
    #cv = int(moments['m01'] / moments['m00'])
    #cnt_point = rs.rs2_deproject_pixel_to_point(intrinsics, [cu, cv], depthimg[cv, cu])
    return ((100*mean_point[0]-hole_position[0])**2 + (100*mean_point[1]-hole_position[1])**2)**0.5


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

def findPresegRois(mask):
    rois = []
    if cv2.__version__ >= "4.0":
        contours, hierar = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _,contours, hierar = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    regions, labeled = assignLabelContour(mask,contours)
    for region in regions:
        cntArray = contourAsArray(region.contour)
        row_min = np.min(cntArray[:,0])
        row_max = np.max(cntArray[:,0])
        col_min = np.min(cntArray[:,1])
        col_max = np.max(cntArray[:,1])
        rois.append([[row_min,row_max],[col_min,col_max]])
    return rois


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
        cones_mask_roi=[mask[485:620,830:970],mask[485:620,660:830],mask[485:620,500:626]]
        cones_depth_roi =[depthimg[485:620,830:970],depthimg[485:620,660:830],depthimg[485:620,500:626]]

    elif parity is "even":
        cones_mask_roi=[mask[485:620, 755:885], mask[485:620, 580:695], mask[485:620, 405:545]]
        cones_depth_roi=[depthimg[485:620, 755:885], depthimg[485:620, 580:695], depthimg[485:620, 405:545]]
    else:
        raise Exception("parity can only be odd or even")

    cones_distance=[]
    for cone_idx in range(3):
        roi = cones_mask_roi[cone_idx]
        cone_depthimg = 100*cones_depth_roi[cone_idx]
        cone_depthimg_aux = np.where(roi==255,cone_depthimg,0)
        cone_depthimg_aux = np.where((48.6>cone_depthimg_aux) & (cone_depthimg_aux>44.16),cone_depthimg_aux,0)
        #plt.imshow(cone_depthimg_aux)
        masked_values = cone_depthimg_aux.astype(np.bool).astype(np.float)
        num_of_valids= masked_values.sum()
        depth_sum = cone_depthimg_aux.sum()
        if num_of_valids > 0:
            distance = depth_sum / num_of_valids
            cones_distance.append(distance)
        else:
            cones_distance.append(46.16)
        #plt.show()
    return cones_distance

def classifySeedling(seedlingArea,margins):
    if seedlingArea >= margins[2]:
        return [1.0]
    elif seedlingArea >= margins[1] and seedlingArea < margins[2]:
        return [2.0]
    elif seedlingArea >= margins[0] and seedlingArea < margins[1]:
        return [3.0]
    else:
        return [0.0]

def checkIfInRange(val,lowerT,upperT):
    if lowerT[0]<=val[0] and val[0]<= upperT[0] and lowerT[1]<=val[1] and val[1]<= upperT[1] and lowerT[2]<=val[2] and val[2]<= upperT[2]:
        return True
    else:
        return False
"""
SLIC-Ellipsoid based segmentation funcions
"""
def checkPointEllipsoid(point,center,radii,rotation,eps=0.01):
    pnt_mod = (np.matmul(rotation, point.reshape(3, 1) - center.reshape(3, 1))).reshape(3,)
    val = ((pnt_mod[0] / radii[0]) ** 2) + ((pnt_mod[1] / radii[1]) ** 2) + ((pnt_mod[2] / radii[2]) ** 2)
    if val ** 0.5 < 1.0 + eps:
        return True
    else:
        return False

def checkPointEllipsoidDict(point,ellip_dict,eps=0.01):
    pnt_mod = (np.matmul(ellip_dict["rot"], point.reshape(3, 1) - ellip_dict["cent"].reshape(3, 1))).reshape(3,)
    val = ((pnt_mod[0] / ellip_dict["rad"][0]) ** 2) + ((pnt_mod[1] / ellip_dict["rad"][1]) ** 2) + ((pnt_mod[2] / ellip_dict["rad"][2]) ** 2)
    if val ** 0.5 < 1.0 + eps:
        return True
    else:
        return False

def findEllipsoid(point,ellips_dict=None,eps=0.01):
    if ellips_dict is None:
        return -1
    else:
        if checkPointEllipsoidDict(point,ellips_dict["bl"],0.15) or checkPointEllipsoidDict(point,ellips_dict["gray"],0.95): #val bl = 0.15
            return "bg"
        elif checkPointEllipsoidDict(point,ellips_dict["br"],0.08): #val br = 0.08
            return "cone"
        elif checkPointEllipsoidDict(point,ellips_dict["dg"],0.7) or checkPointEllipsoidDict(point,ellips_dict["lg"],0.8):
            return "seedling"
        else:
            return False


class seedlingClassifier():
    def __init__(self,intrinsics):
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
        self.seedlingClassifierModel = None
        self.colorSegmentationModel = None
        self.row_roi = 450
        self.col_roi = [360, 1200]
        self.hole_positions = [[-8.91548333333333,12.2185666666667],[-4.349625,11.0752125],[0.362765555555556,10.9953444444444],[4.84777,11.11838],[9.30483333333333,11.2195888888889],[14.1567625,11.200375]]
        self.intrinsics = intrinsics
        self.depth_scale = 9.999999747378752e-05
        self.ellipsoids_dict = None
    def modbusConnect(self,modbusClient): #### I need to know if I'm connected to the server.
        self.modbusClient = modbusClient
        if self.modbusClient.is_socket_open() is True:
            print("Seedling classifier's connection to modbus server -> successful")
            self.modbusConnectedFlag = True
        else:
            self.modbusConnectedFlag = False
            print("Seedling classifier's connection to modbus server -> failed")

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
        _stime = time()
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
        #self.__spatial_filter = rs.spatial_filter()
        #self.__spatial_filter.set_option(rs.option.holes_fill, 3)
        sleep(0.06)
        self.__rspipeline.stop()
        sleep(0.06)
        # Second start
        self.__rspipeline_profile = self.__rspipeline.start(self.__rsconfig)
        self.__rsdepth_sensor = self.__rspipeline_profile.get_device().first_depth_sensor()
        self.__rsdepth_sensor.set_option(rs.option.emitter_enabled, 1)
        self.__rsdepth_sensor.set_option(rs.option.laser_power, 250)
        self.__rsdepth_sensor.set_option(rs.option.depth_units, 0.0001)
        self.__temp_filter = rs.temporal_filter()
        self.__temp_filter.set_option(rs.option.filter_smooth_alpha, 0.7)
        self.__temp_filter.set_option(rs.option.filter_smooth_delta, 10)
        self.__temp_filter.set_option(rs.option.holes_fill, 1.0)
        #self.__spatial_filter = rs.spatial_filter()
        #self.__spatial_filter.set_option(rs.option.holes_fill, 3)
        frames = self.__rspipeline.wait_for_frames(timeout_ms=2000)
        aligned_frames = self.__align.process(frames)
        for i in range(self.initial_discard_frames):
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            #depth_frame = self.__spatial_filter.process(depth_frame)
            depth_frame = self.__temp_filter.process(depth_frame)
        self.cameraInitializedFlag = True
        print("Camera initialization time: {:2.3f}".format(time()-_stime))
    def cameraRestart(self):
        if self.__rspipeline is not None:
            try:
                self.cameraInitialize()
            except Exception as e:
                print("Camera not found: {}".format(e))
    def getImages(self,mode):
        _stime = time()
        if mode is "online":
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
                    #depth_frame = self.__spatial_filter.process(depth_frame)
                    depth_frame = self.__temp_filter.process(depth_frame)
                depth_frame = self.__temp_filter.process(depth_frame)
                color_frame = aligned_frames.get_color_frame()  # From frames.get_color_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                self.depthImg = self.depth_scale*depth_image
                self.rgbImg = color_image
                print("Imaging time: {:2.3f}".format(time() - _stime))
                return True
            except Exception as e: #<-I have to check this
                print("Problem while getting images: {}".format(e))
                return False
        elif mode is "offline":
            return True
        else:
            raise Exception("Error during obtaining images: mode not specified")


    def correctZValues(self,conedistances):
        z1correction = (conedistances[0] - 46.16) * 10
        z2correction = (conedistances[1] - 46.16) * 10
        z3correction = (conedistances[2] - 46.16) * 10
        self.modbusClient.writeZcorrection(z1correction,z2correction,z3correction)

    def classifySeedlingArea(self,leaf_area):
        return self.seedlingClassifierModel.predict(np.array([leaf_area]).reshape(1,1))[0]


    def onlysegmentation(self,mode="offline"):
        if self.cameraInitializedFlag is True or mode is "offline":
            __processing_start = time()
            if self.getImages(mode) is True:
                _stime = time()
                rgbGUI = self.rgbImg.copy()
                rgb_padded = np.zeros(self.rgbImg.shape, dtype=np.uint8)
                rgb_padded[self.row_roi:-1, self.col_roi[0]:self.col_roi[1]] = self.rgbImg[self.row_roi:-1, self.col_roi[0]:self.col_roi[1]]
                preseg_rgb = np.zeros(self.rgbImg.shape, dtype=np.uint8)
                mask_depth = np.zeros(self.rgbImg.shape[0:2], dtype=np.uint8)
                mask_seedlings = np.zeros(self.rgbImg.shape[0:2], dtype=np.uint8)
                mask_cones = np.zeros(self.rgbImg.shape[0:2], dtype=np.uint8)

                ##Get the predefined ROI
                depth_roi = self.depthImg[self.row_roi:, self.col_roi[0]:self.col_roi[1]]
                rgb_roi = rgb_padded[self.row_roi:, self.col_roi[0]:self.col_roi[1]]

                ##Segmentation using depth
                mask_depth_roi = np.where((depth_roi < 0.47) & (depth_roi > 0.28), 255, 0).astype(np.uint8)  # pixels between 3cm and 33 cm
                preseg_rgb_roi = cv2.bitwise_and(rgb_roi, rgb_roi, mask=mask_depth_roi)
                mask_depth[self.row_roi:, self.col_roi[0]:self.col_roi[1]] = mask_depth_roi
                preseg_rgb[self.row_roi:, self.col_roi[0]:self.col_roi[1]] = preseg_rgb_roi
                preseg_roi_ranges = findPresegRois(mask_depth)

                ##Segmentation using color
                for idx,roi in enumerate(preseg_roi_ranges):
                    ltime = localtime()
                    name = "../subimages/IMG_{}_{}_{}_{}.png".format(ltime.tm_hour, ltime.tm_min, ltime.tm_sec,idx)
                    if roi[0][1]>roi[0][0] and roi[1][1]>roi[1][0]:
                        #cv2.imwrite(name,preseg_rgb[roi[0][0]:roi[0][1],roi[1][0]:roi[1][1]])
                        img_roi_yuv = cv2.cvtColor(preseg_rgb[roi[0][0]:roi[0][1],roi[1][0]:roi[1][1]],cv2.COLOR_BGR2YUV)
                        img_roi_hsv = cv2.cvtColor(preseg_rgb[roi[0][0]:roi[0][1],roi[1][0]:roi[1][1]],cv2.COLOR_BGR2HSV_FULL)
                        #pixel_num = img_roi_yuv.shape[0] * img_roi_yuv.shape[1]
                        pixel_num = np.sum(np.where(mask_depth[roi[0][0]:roi[0][1],roi[1][0]:roi[1][1]]>0,1,0))
                        nsp = int(0.007 * pixel_num) # img_roi_yuv.shape[0] * img_roi_yuv.shape[1] Superpixels number = 0.3% Total pixel number
                        if nsp>4:
                            labeled = slic(img_roi_yuv, n_segments=nsp, start_label=1,sigma=1,mask=mask_depth[roi[0][0]:roi[0][1],roi[1][0]:roi[1][1]])
                            mask_seedlings_aux = np.zeros(img_roi_yuv.shape[0:2], dtype=np.uint8)
                            mask_cones_aux = np.zeros(img_roi_yuv.shape[0:2], dtype=np.uint8)
                            for label in range(1, np.max(labeled) + 1):
                                selected_pixels = img_roi_yuv[labeled == label]
                                #selected_pixels = img_roi_hsv[labeled == label]
                                mean = (np.mean(selected_pixels, axis=0).reshape((1,1,3))).astype(np.uint8)
                                #mean = cv2.cvtColor(cv2.cvtColor(mean,cv2.COLOR_YUV2BGR),cv2.COLOR_BGR2HSV_FULL)
                                mean = mean.reshape(3)
                                #ellipsoid_val = findEllipsoid(mean, self.ellipsoids_dict, eps=0.6)
                                #if  ellipsoid_val is "seedling":
                                #    mask_seedlings_aux = np.where(labeled == label, 255, mask_seedlings_aux).astype(np.uint8)
                                #elif ellipsoid_val is "cone":
                                #    mask_cones_aux = np.where(labeled == label, 255, mask_cones_aux).astype(np.uint8)
                                #   checkIfInRange(mean,[48,25,60],[120,255,240])
                                if self.colorSegmentationModel.predict(mean.reshape((-1,3))) == 1.0: #[35,30,60],[139,255,250])  checkIfInRange(mean,[25,10,60],[170,255,251])
                                    mask_seedlings_aux = np.where(labeled == label, 255, mask_seedlings_aux).astype(np.uint8)
                            mask_seedlings[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]] = mask_seedlings_aux
                            mask_cones[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]] = mask_cones_aux
                mask_seedlings = removeSmallRegions(mask_seedlings,RegionThresh=800)
                print("Segmentation time: {:2.3f}".format(time() - _stime))
                return mask_seedlings,mask_cones
            else:
                self.rgbImg = None
                self.depthImg = None
                print("Couldn't get the images")
                return None
        else:
            print("Initialize the camera first")
            return None

    def processSeedlings(self,seedlingParity,mode="offline"):
        __processing_start = time()
        _stime = time()
        mask_seedlings,mask_cones = self.onlysegmentation(mode)
        rgbGUI = self.rgbImg.copy()
        #Obtain camera-cones distances
        cone_distances = estimate_cones_distances(mask_cones, self.depthImg, seedlingParity)
        ##Obtain contours
        if cv2.__version__ >= "4.0":
            contours, hierar = cv2.findContours(mask_seedlings, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _,contours, hierar = cv2.findContours(mask_seedlings, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours,hierar = remove_small_contours(contours,hierar, 45)

        ## ASSIGN REGIONS AND CONTOURS TO SEEDLING HOLES AND CLASSIFY THEM
        S0, S1, S2 = assign_to_seedling2(mask_seedlings, contours,hierar, self.depthImg, self.hole_positions, 6.9,seedlingParity,self.intrinsics,cone_distances)

        print("Area estimation time: {:2.3f}".format(time() - _stime))

        if cv2.__version__ >= "4.0":
            contours, hierar = cv2.findContours(mask_seedlings, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            _,contours, hierar = cv2.findContours(mask_seedlings, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #contours, hierar = remove_small_contours(contours, hierar, 45)

        cv2.drawContours(rgbGUI, contours, -1, [100, 60, 200], 2)

        """
        ##This block could be used in the case there exists any machine learning based classifier model
        
        q0 = self.seedlingClassifierModel.predict([[S0.area, S0.height]])
        q1 = self.seedlingClassifierModel.predict([[S1.area, S1.height]])
        q2 = self.seedlingClassifierModel.predict([[S2.area, S2.height]])
        """

        #Classify seedlings according to their leaf area
        leaf_area_margins = [5.0,13.8,19.89]
        if self.seedlingClassifierModel is None:
            raise Exception("ERROR: Seedling LinearSVC model wasn't specified")
        _stime = time()
        q0 = self.classifySeedlingArea(S0.area) #classifySeedling(S0.area,leaf_area_margins)
        q1 = self.classifySeedlingArea(S1.area) #classifySeedling(S1.area,leaf_area_margins)
        q2 = self.classifySeedlingArea(S2.area) #classifySeedling(S2.area,leaf_area_margins)
        print("Classification time: {:2.3f}".format(time() - _stime))

        cv2.rectangle(rgbGUI, *S0.enclosingBox, [255, 0, 0], 1)
        cv2.rectangle(rgbGUI, *S1.enclosingBox, [255, 0, 0], 1)
        cv2.rectangle(rgbGUI, *S2.enclosingBox, [255, 0, 0], 1)
        print("S0: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}, Cone distance = {}".format(S0.area, S0.height,q0,cone_distances[0]))
        print("S1: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}, Cone distance = {}".format(S1.area, S1.height,q1,cone_distances[1]))
        print("S2: Area = {:3.3f} cm\u00b2, Average Height= {:3.3f} cm, quality? = {}, Cone distance = {}".format(S2.area, S2.height,q2,cone_distances[2]))
        print("Processing Time: {} seconds".format(time() - __processing_start))
        if self.modbusConnectedFlag == True:
            __sending_start = time()
            print("Sending results to the server ...")
            self.writeSeedlingsQuality(int(q0),int(q1),int(q2))#self.writeSeedlingsQuality(int(q0),int(q1[0]),int(q2[0]))
            self.correctZValues(cone_distances)
            self.modbusClient.cvFinishProcessing()
            print("Results sent to server. Sending time: {} seconds \n".format(time()-__sending_start))
        return rgbGUI

