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
from paho.mqtt.client import Client as mqttClient
from glob import glob
import warnings
warnings.filterwarnings("ignore")


sys.path.insert(1,"../")
from modbus_mqtt.libseedlingmodbus import SeedlingModbusClient
from modbus_mqtt import libseedlingmodbus as lsmodb
from common_functions import *
from Ericks_system import ericks_functions

#GUI and mouse
#cv2.namedWindow("Results")
#cv2.setMouseCallback("Results", click_depth_rgb)

#SOME GLOBAL VARIABLES AND CONSTANTS
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

#INITIALIZE COMPUTER VISION SYSTEMS
#Paulo's CV
cvSystem = seedlingClassifier(intrinsics)
file = open("leaf_area_seedling_classifier.pkl", "rb")
cvSystem.seedlingClassifierModel = pickle.load(file)
file.close()
file = open("color_segmentation_svc.pkl", "rb")
cvSystem.colorSegmentationModel = pickle.load(file)
file.close()

#files = glob("../datasets/plantines-16-06-21-erick"+"/Depth*.jpg")
folder = "/home/amaranth/Desktop/Robot_UPAO/new_dataset_20_09_2021/bandeja_6_17_09_2021"
files = glob("{}/*.png".format(folder))
files.sort()

for idx,file in enumerate(files):
    print(file)
    cvSystem.rgbImg = cv2.imread(file, 1)
    #cvSystem.depthImg = depth_scale*np.load(file[0:-3] + "npy")
    cvSystem.depthImg = np.load(file[0:-3] + "npy")
    if idx%2 is 0:
        print("odd")
        rgbGUI = cvSystem.processSeedlings("odd", "offline")
    else:
        print("even")
        rgbGUI = cvSystem.processSeedlings("even", "offline")
    cv2.imshow("results",rgbGUI)
    cv2.waitKey(0)
cv2.destroyAllWindows()
