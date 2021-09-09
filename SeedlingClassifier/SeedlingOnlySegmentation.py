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
from matplotlib import pyplot as plt #JUST TO SEE THE GRADIENT
from glob import glob
import warnings
warnings.filterwarnings("ignore")


sys.path.insert(1,"../")
from modbus_mqtt.libseedlingmodbus import SeedlingModbusClient
from modbus_mqtt import libseedlingmodbus as lsmodb
from common_functions import *
from Ericks_system import ericks_functions


def on_connect_(self, userdata, flags, rc):
    print("Connected MQTT with result code " + str(rc))
    main_mqtt_client.subscribe("PUT_HERE_THE_TOPIC_OF_INTEREST_tinterest")

def on_message_(self, userdata, msg):
    topic = str(msg.topic)
    payload = str(msg.payload.decode('utf-8'))
    print("payload: " + payload)
    print("topic: " + topic)
    global CV_system_switch
    if topic is "SysP":
        CV_system_switch = "SysP"
        print("CV system switched to Paulo's system")
    elif topic is "SysE":
        print("CV system switched to Erick's system")
        CV_system_switch = "SysE"
    else:
        print("CV system switched to default")
        CV_system_switch = "SysP"

def click_labeled(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Label: {}".format(labeled_full[y,x])) #
        print("Distances: {}".format(dists_full[y,x]))
        #print("Distances: {}.".format(kmeans_model.transform(colordata.reshape(1,-1))))
    return


cv2.namedWindow("segmented")
cv2.setMouseCallback("segmented", click_labeled)

args = sys.argv

if "-debug" in args:
    print("Entering debug mode ...")

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
CV_system_switch = "SysP"
#ODD_RGB = cv2.imread("Offline_files/IMG_17_50_40.jpg",cv2.IMREAD_COLOR)
#ODD_DEPTH = np.load("Offline_files/IMG_17_50_40.npy")
ODD_RGB = cv2.imread("../datasets/seedlings_18_06_2021/IMG_15_5_36.jpg",cv2.IMREAD_COLOR)
ODD_DEPTH = np.load("../datasets/seedlings_18_06_2021/IMG_15_5_36.npy")
EVEN_RGB = cv2.imread("Offline_files/IMG_15_38_14.jpg",cv2.IMREAD_COLOR)
EVEN_DEPTH = np.load("Offline_files/IMG_15_38_14.npy")
CV_MODE = "offline"

## OPEN MODELS
#Paulo's CV system related models
#file = open("../Deprecated/kmeans_model_v0.5.pkl", "rb")
#segmentation_model = pickle.load(file)
#file2 = open("../Deprecated/Seedling_Classifier_model.pkl", "rb")
#seedling_classifier = pickle.load(file2)
#Erick's CV system related models


args = sys.argv

modServerIp = "192.168.1.103"
modServerPort = 5030
mqttBrokerIp = "192.168.1.103"
mqttBrokerPort = 1883

if "-serverIp" in args:
    idx = args.index("-serverIp")
    try:
        modServerIp = args[idx+1]
    except:
        raise Exception("Server's IP wasn't specified")

if "-serverPort" in args:
    idx = args.index("-serverPort")
    try:
        modServerPort = int(args[idx+1])
    except:
        raise Exception("Server's Port wasn't specified or is not valid" )

if "-brokerIp" in args:
    idx = args.index("-brokerIp")
    try:
        modServerIp = args[idx+1]
    except:
        raise Exception("Broker's IP wasn't specified")

if "-brokerPort" in args:
    idx = args.index("-brokerPort")
    try:
        modServerPort = int(args[idx+1])
    except:
        raise Exception("Broker's Port wasn't specified or is not valid" )

#INITIALIZE MAIN MQTT CLIENT
main_mqtt_client = mqttClient()
#try:
#    if main_mqtt_client.connect(mqttBrokerIp,mqttBrokerPort) is True:
#        print("MQTT connection -> Successful")
#    else:
#        print("MQTT connection -> Failed")
#except:
#    print("MQTT connection -> Failed")

if main_mqtt_client.is_connected():
    main_mqtt_client.on_connect = on_connect_
    main_mqtt_client.on_message = on_message_

modbusClient = SeedlingModbusClient(modServerIp,modServerPort)
modbusClientConnectedFlag = modbusClient.connectToServer()
if modbusClientConnectedFlag is True:
    print("Modbus Client's connection -> successful")
else:
    print("Modbus Client's connection -> failed")

#INITIALIZE COMPUTER VISION SYSTEMS
#Paulo's CV
cvSystem = seedlingClassifier(intrinsics)
#cvSystem.initial_centroids = np.load("../Deprecated/initial_centroids.npy")
#cvSystem.initial_centroid_idx = {"seedlings":[2,3,4,6,8,9,10,11,12,13],"cones":[5,7,14],"bg":[0,1]}
with open("colors_ellipsoids_dict.pkl","rb") as file:
    cvSystem.ellipsoids_dict = pickle.load(file)
#cvSystem.modbusConnect(modbusClient)
#if CV_MODE is "online":
#    cvSystem.cameraInitialize()
#Erick's CV
#cvSystem2 = ericks_functions.ErickSeedlingClassifier(modbusClient)


#if modbusClientConnectedFlag is True:
#    modbusClient.writeCvStatus(lsmodb.CV_WAITING_STAT)
#plcInstruction = lsmodb.PLC_PROCODD_INST

#files = glob("../datasets/plantines-16-06-21-erick"+"/Depth*.jpg")
folder = "/home/amaranth/Desktop/Robot_UPAO/Seedling_vision/datasets/seedling_dataset_02_06_2021_21"
files = glob("{}/*.jpg".format(folder))
#print("../datasets/plantines-16-06-21-erick/Depth"+files[1][45:-3]+"npy")
#num = 0
cvSystem.rgbImg = cv2.imread("../datasets/images_paper/Sample1.jpg",1)
cvSystem.depthImg = cvSystem.depth_scale*np.load("../datasets/images_paper/Sample1.npy")
seedling_mask,cones_mask= cvSystem.onlysegmentation()
segmented = cv2.bitwise_and(cvSystem.rgbImg,cvSystem.rgbImg,mask=seedling_mask)
cv2.imshow("original",cvSystem.rgbImg)
cv2.imshow("segmented", segmented)
cv2.imshow("colorized",colorizeDepth(cvSystem.depthImg,0.28,0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
#segmented,mask = cvSystem.onlysegmentation()
#cv2.imshow("mask",segmented)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
"""for file in files:
    print(file)
    cvSystem.rgbImg = cv2.imread(file, 1)
    #cvSystem.depthImg = depth_scale*np.load(file[0:-3] + "npy")
    cvSystem.depthImg = np.load(file[0:-3] + "npy")
    print(len(file[54:-3]))
    seedling_mask,cones_mask= cvSystem.onlysegmentation()
    segmented = cv2.bitwise_and(cvSystem.rgbImg,cvSystem.rgbImg,mask=seedling_mask)
    #cv2.imwrite(file[0:-4]+"_mask.png",seedling_mask)
    cv2.imshow("original",cvSystem.rgbImg)
    cv2.imshow("segmented", segmented)
    cv2.imshow("colorized",colorizeDepth(cvSystem.depthImg,0.28,0.5))
    cv2.waitKey(0)
cv2.destroyAllWindows()
"""