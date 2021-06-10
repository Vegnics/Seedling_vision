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
from common_functions import *


args = sys.argv

if "-debug" in args:
    print("Entering debug mode ...")

#GUI and mouse
cv2.namedWindow("Results")
#cv2.setMouseCallback("Results", click_depth_rgb)

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


## OPEN MODELS
file = open("MLmodel_v0.3.pkl", "rb")
segmentation_model = pickle.load(file)
file2 = open("Seedling_Classifier_model.pkl", "rb")
seedling_classifier = pickle.load(file2)

#INITIALIZE COMPUTER VISION SYSTEM
cvSystem = seedlingClassifier(segmentation_model,seedling_classifier,intrinsics)
cvSystem.modbusConnect("192.168.1.103",502)
cvSystem.cameraInitialize()

if cvSystem.modbusConnectedFlag is True:
    cvSystem.modbusClient.writeCvStatus(lsmodb.CV_WAITING_STAT)
plcInstruction = lsmodb.PLC_PROCODD_INST

while True:
    if cvSystem.modbusConnectedFlag is True:
        plcInstruction = cvSystem.modbusClient.getPLCInstruction()
    if plcInstruction == lsmodb.PLC_PROCODD_INST:
        if cvSystem.modbusConnectedFlag is True:
            cvSystem.modbusClient.writeCvStatus(lsmodb.CV_PROCESSING_STAT)
            """
            cvSystem.getImages()
            depth = cvSystem.depthImg
            color = cvSystem.rgbImg
            cv2.imshow("Results",color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            rgbGUI = cvSystem.processSeedlings("odd") # IM CHANGING CONSTANTLY THIS PARAMETER
            #SAVE IMAGES
            #ltime = localtime()
            #name = "seedling_dataset_02_06_2021_21/IMG_{}_{}_{}".format(ltime.tm_hour,ltime.tm_min,ltime.tm_sec)
            #print(name)
            #cv2.imwrite(name+".jpg",cvSystem.rgbImg)
            #np.save(name+".npy",cvSystem.depthImg)

    elif plcInstruction == lsmodb.PLC_PROCEVEN_INST:
        if cvSystem.modbusConnectedFlag is True:
            cvSystem.modbusClient.writeCvStatus(lsmodb.CV_PROCESSING_STAT)
            rgbGUI = cvSystem.processSeedlings("even")
            # SAVE IMAGES
            #ltime = localtime()
            #name = "seedling_dataset_02_06_2021_21/IMG_{}_{}_{}".format(ltime.tm_hour, ltime.tm_min, ltime.tm_sec)
            #print(name)
            #cv2.imwrite(name + ".jpg", cvSystem.rgbImg)
            #np.save(name + ".npy", cvSystem.depthImg)
    try:
        cv2.imshow("Results",rgbGUI)
        cv2.waitKey(4500)
        cv2.destroyAllWindows()
        finished = True
    except:
        pass
