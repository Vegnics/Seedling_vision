import numpy as np
import cv2
#import time
from time import time,sleep

from datetime import datetime
#import shutil
import json
import paho.mqtt.client as mqtt
import os
import imageio

import pyrealsense2 as rs
import pickle
from sklearn.cluster import KMeans

import sys
from threading import Timer,Thread

sys.path.insert(1,"../")
from modbus_mqtt.libseedlingmodbus import SeedlingModbusClient
from modbus_mqtt import libseedlingmodbus as lsmodb

##########VARIABLES#############
data = []
labels = []

plantpath = '/home/erickmfs/ai_apps/seedling_groups/darknet-yolov3/images/test_images/' #'/home/plantinator/plantinator/images/test_images'
seedldestpath = '/home/erickmfs/ai_apps/seedling_groups/darknet-yolov3/images/seedlings/' #'/home/root/vision'
modelpath = "/home/erickmfs/ai_apps/seedling_groups/darknet-yolov3/" #'/home/plantinator/plantinator'

MQTT_SERVER = "localhost" #"192.168.0.8" #"localhost"

MQTT_PATH = "cts/#"
NOMBRE_ESCLAVO = "cts"
COSTO_KWH = 0.6
working_mode = 0 #if 0: control mode, if 1: manual mode

topic4listenig = "robot/vision/ia/#"
topic_prueba = "robot/vision/ia/prueba"
topic_parametros = "robot/vision/ia/parametros"
topic_ima1_ruta = "robot/vision/imagen1/ruta"
topic_ima1_cali = "robot/vision/imagen1/calidad"
topic_ima2_ruta = "robot/vision/imagen2/ruta"
topic_ima2_cali = "robot/vision/imagen2/calidad"
topic_ima3_ruta = "robot/vision/imagen3/ruta"
topic_ima3_cali = "robot/vision/imagen3/calidad"

#limits for every group of seedling predictions
g3min = 60
g3max = 175
g2min = 175
g2max = 275
g1min = 275
g1max = 400
hmin = 194
hmax = 232
score_threshold = 0.98
left_threshold = 2614
right_threshold = 4676

DIAG_CMD = "DIAG"
user = "cts"
password = "123456789cts"

def class_pred_in_letters(pred):  # from number-based seedling classification to letters
    if pred == 3:
        return "A"
    if pred == 2:
        return "B"
    if pred == 1:
        return "C"


def classify_pred(box, score):  # classifications images according to calibrated thresholds
    print("Classifying predictions done by yolov3")
    x, y, w, h = box
    area = w * h
    if score < score_threshold:  # It is suspected that it is not average seedling
        # then it is necessary to know if it is bad or good seedling
        if area < left_threshold:
            # bad seedling
            return 1  # 0
        if area > right_threshold:
            # good seeling
            return 3  # 2
        else:
            return 2  # 1  #TO BE CONSIDERED!
    else:
        # at least that it has a big area
        if area > left_threshold and area < right_threshold:  # OJOOOO!! take it into account
            # average seedling
            return 2  # 1
        if area > right_threshold:
            # good seedling
            return 3  # 2
        else:
            return 1  # 0
    # print("no pasa")


def cropimage(image, x_min, x_max, y_min, y_max, res_per):  # resizing images
    # per_res: resizing percentage
    # image = imageio.imread(path)
    # image = np.array(image)
    # crop the image using array slices -- it's a NumPy array
    # after all!
    cropped = image[y_min:y_max, x_min:x_max]  # 443:720, 403:1082]
    size = (int(cropped.shape[1] * res_per), int(cropped.shape[0] * res_per))  # image size proportion 720/1424 = 0.51
    # resize image
    # rsimg = cv2.resize(image,size) #this is really important to facilitate seedling manipulation
    cropped = cv2.resize(cropped, size)
    return np.array(cropped)


def imageremix(base_img, mask_img, rv, cv):  # Put two images together
    print("Mixing 2 images")
    brv = rv  # base image row value
    bcv = cv
    bimg = np.array(base_img)
    mimg = np.array(mask_img)
    mimg_r, mimg_c, _ = mimg.shape
    bimg_r, bimg_c, _ = bimg.shape
    limitflag_c = True
    # limitflag_r = True

    # if debug_mode:
    #    print("mask image row and column: " + str(mimg_r) + ", " + str(mimg_c))
    #    print("base image row and column: " + str(bimg_r) + ", " + str(bimg_c))

    for r in range(mimg_r):
        for c in range(mimg_c):
            if mimg[r, c, 0] > 0:  # different from zero
                if limitflag_c:
                    bimg[brv, bcv, :] = mimg[r, c, :]
            bcv += 1
            if bcv >= bimg_c:
                limitflag_c = False
            else:
                limitflag_c = True
        brv += 1
        bcv = cv
        if brv >= bimg_r: break

    return bimg


def get_prediction(modelpath, image_input):
    print("begin PREDICTION PROCESS")
    weight_path = os.path.join(modelpath, 'yolov3_training_last.weights')
    cfg_path = os.path.join(modelpath, 'yolov3_testing.cfg')
    net = cv2.dnn.readNet(weight_path, cfg_path)
    print("yolov3 loaded")
    classes = []
    with open(modelpath+"classes.txt", "r") as f:
        classes = f.read().splitlines()
    # cap = cv2.VideoCapture('video4.mp4')
    # cap = 'test_images/<your_test_image>.jpg'
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    ########################################################
    img = cv2.imread(image_input)  # "test_images/img19.png")
    ########################################################

    height, width, _ = img.shape
    if width > 416:
        img = cropimage(img, 403, 1082, 443, 720, 0.5625)  # according to 416x146pixel image size input for yolov3 model
        img = imageremix(imageio.imread(os.path.join(modelpath, "size_ref.jpg")), img, int(679 / 2 - 416 / 2),
                         int(416 / 2 - 277 / 2))
        height, width, _ = img.shape

    print("height: " + str(height) + ", width: " + str(width))

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    print("aquiiiiii")

    boxes = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
    confidences = [0, 0, 0]
    class_ids = [0, 0, 0]

    PM = np.ones([10, 3, 6])  # *-1 #Position matrix
    idx = [0, 0, 0]  # np.zeros([1,3]) #with zeros considering 0 position at the beginning

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if center_x > g1min and center_x <= g1max and center_y >= hmin and center_y <= hmax:
                    # 1st group
                    PM[idx[0], 0, :] = [classify_pred([x, y, w, h], confidence), x, y, w, h, confidence]
                    idx[0] += 1
                if center_x > g2min and center_x <= g2max and center_y >= hmin and center_y <= hmax:
                    # 2nd group
                    PM[idx[1], 1, :] = [classify_pred([x, y, w, h], confidence), x, y, w, h, confidence]
                    idx[1] += 1
                if center_x > g3min and center_x <= g3max and center_y >= hmin and center_y <= hmax:
                    # 3th group
                    PM[idx[2], 2, :] = [classify_pred([x, y, w, h], confidence), x, y, w, h, confidence]
                    idx[2] += 1
        print("por aquiiiii")

        # here it is necessary to treat boxes, confidences and class_ids as stablished array size variables
        # in order to know if there is a cell with no seedling detection

        for pidx in range(3):
            val = -1
            print(idx[pidx])
            if idx[
                pidx] > 0:  # if there is -1 value at position it must be recognized as no seedling detection situation
                for i in range(idx[pidx]):
                    print("i: ", i)
                    c = PM[
                        i, pidx, 0]  # this stage goes for discrimination in case more than one seedling was detected and classified at the same cell
                    if c > val:  # the greatest class is our guy!
                        indx = i
                        val = c

                #################those variables could be initialized as -1 arrays################3
                print("PM: ", PM[indx, pidx, 1:5])
                boxes[pidx] = PM[indx, pidx, 1:5]
                confidences[pidx] = float(PM[indx, pidx, 5])
                class_ids[pidx] = PM[indx, pidx, 0]
                # boxes.append(PM[indx,pidx,1:5])
                # confidences.append(float(PM[indx,pidx,5]))
                # class_ids.append(PM[indx,pidx,0])

            # else:
            #    boxes.append((0,0,0,0))
            #    confidences.append(0)
            #    class_ids.append(0)

            print("por aquiiiii 2")

        # boxes.append([x, y, w, h])
        # confidences.append((float(confidence)))
        # class_ids.append(class_id)

    print("saliooo")
    print("shape boxes: ", len(boxes))
    print(boxes[0])
    print(boxes[1])
    print(boxes[2])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    print("indexes: ", indexes)
    seedlfnames = [" ", " ", " "]

    # This is just for saving seedling images each one in different file
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            print("paso aquiiii")
            # label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            # color = colors[i]
            # cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            # cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)

            crop_img = img[int(y):int(y + h + 40), int(x):int(x + w + 40)]
            now = datetime.now()
            dt_str = now.strftime("%d-%m-%Y_%H-%M-%S.%f")
            sfn = 'seedling_' + dt_str + '.png'
            dpath = os.path.join(seedldestpath, sfn)
            cv2.imwrite(dpath, crop_img)
            seedlfnames[i] = sfn
            # seedlfnames.append(sfn)

            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(0)

    # seedlfnames = np.array(seedlfnames)
    return seedlfnames, class_ids


def start_processing_image(img):
    print("Processing image")
    sn, sc = get_prediction(modelpath, img)
    return sn, sc


class ErickSeedlingClassifier(mqtt.Client):
    def __init__(self, modbusClient):
        mqtt.Client.__init__(self)
        self.topic = None
        self.payload = None
        self.modbusclient = modbusClient
        self.g3min = 60
        self.g3max = 175
        self.g2min = 175
        self.g2max = 275
        self.g1min = 275
        self.g1max = 400
        self.hmin = 194
        self.hmax = 232
        self.score_threshold = 0.98
        self.left_threshold = 2614
        self.right_threshold = 4676
        self.topic4listenig = "robot/vision/ia/#"
        self.topic_prueba = "robot/vision/ia/prueba"
        self.topic_parametros = "robot/vision/ia/parametros"
        self.topic_ima1_ruta = "robot/vision/imagen1/ruta"
        self.topic_ima1_cali = "robot/vision/imagen1/calidad"
        self.topic_ima2_ruta = "robot/vision/imagen2/ruta"
        self.topic_ima2_cali = "robot/vision/imagen2/calidad"
        self.topic_ima3_ruta = "robot/vision/imagen3/ruta"
        self.topic_ima3_cali = "robot/vision/imagen3/calidad"
        self.MQTT_PATH = "cts/#"
        self.plantpath = '../Ericks_system/images/'  # '/home/plantinator/plantinator/images/test_images'
        #self.seedldestpath = '/home/erickmfs/ai_apps/seedling_groups/darknet-yolov3/images/seedlings/'  # '/home/root/vision'
        self.modelpath = "../Ericks_system/"  # '/home/plantinator/plantinator'
        #self.modbusclient.writeCvStatus(lsmodb.CV_WAITING_STAT)
        #self.thread = Timer(0.2, self.check_plc)
        #self.thread.start()

    def getSegmentedImage(self,segmentedImage):
        now = datetime.now()
        dt_str = now.strftime("%d-%m-%Y_%H-%M-%S.%f")
        sfn = 'seedlings_' + dt_str + '.png' ###TIENE QUE SER EN PNG?????
        dpath = os.path.join(self.plantpath, sfn)
        cv2.imwrite(dpath,segmentedImage)
        sleep(0.5)
        print(sfn)
        return dpath

    def processImage(self,segmentedImage):
        imgpath = self.getSegmentedImage(segmentedImage)
        try:
            [seedlpath, seedlquality] = self.start_processing_image(imgpath)
            #self.publish(topic_ima1_ruta, seedlpath[0])  # "plantin1.png")
            #self.publish(topic_ima1_cali, class_pred_in_letters(seedlquality[0]))  # "Calidad A")
            #self.publish(topic_ima2_ruta, seedlpath[1])  # "plantin2.png")
            #self.publish(topic_ima2_cali, class_pred_in_letters(seedlquality[1]))  # "Calidad B")
            #self.publish(topic_ima3_ruta, seedlpath[2])  # "plantin3.png")
            #self.publish(topic_ima3_cali, class_pred_in_letters(seedlquality[2]))  # "Calidad A")
            print("qualities: {}".format(seedlquality))
            if self.modbusclient.is_socket_open():
                self.modbusclient.writeSeedling1Quality(int(seedlquality[0]))  # +1)) #lsmodb.QTY_A) # Define seedling 1 as an A-QUALITY one
                self.modbusclient.writeSeedling2Quality(int(seedlquality[1]))  # +1)) #lsmodb.QTY_A)  # Define seedling 2 as an A-QUALITY one
                self.modbusclient.writeSeedling3Quality(int(seedlquality[2]))  # +1)) #lsmodb.QTY_A)  # Define seedling 3 as an A-QUALITY one
                self.modbusclient.cvFinishProcessing()  # Tell the PLC the processing has finished
        except:
            print(imgpath)
            raise Exception("NAO DEU CERTINHO")
    """    
    def check_plc(self):
        #print(self.modbusclient.getPLCInstruction())
        #plcinstruction = lsmodb.PLC_PROCEVEN_INST #plcinstruction = self.modbusclient.getPLCInstruction()
        if (self.modbusclient.getPLCInstruction() == lsmodb.PLC_PROCEVEN_INST) | (self.modbusclient.getPLCInstruction() == lsmodb.PLC_PROCODD_INST):
            #if(plcinstruction == lsmodb.PLC_PROCEVEN_INST)
            ### run classification
            self.modbusclient.writeCvStatus(
                lsmodb.CV_PROCESSING_STAT)  # Tell the PLC you've received the instruction and you're processing.
            imgname = capture_and_segment_im()  # "img53.png" #capture_and_segment_im() #"img34.png"
            try:
                [seedlpath, seedlquality] = start_processing_image(os.path.join(self.plantpath, imgname))
                self.publish(topic_ima1_ruta, seedlpath[0])  # "plantin1.png")
                self.publish(topic_ima1_cali, class_pred_in_letters(seedlquality[0]))  # "Calidad A")
                self.publish(topic_ima2_ruta, seedlpath[1])  # "plantin2.png")
                self.publish(topic_ima2_cali, class_pred_in_letters(seedlquality[1]))  # "Calidad B")
                self.publish(topic_ima3_ruta, seedlpath[2])  # "plantin3.png")
                self.publish(topic_ima3_cali, class_pred_in_letters(seedlquality[2]))  # "Calidad A")
                print("qualities: {}".format(seedlquality))
                self.modbusclient.writeSeedling1Quality(
                    int(seedlquality[0]))  # +1)) #lsmodb.QTY_A) # Define seedling 1 as an A-QUALITY one
                self.modbusclient.writeSeedling2Quality(
                    int(seedlquality[1]))  # +1)) #lsmodb.QTY_A)  # Define seedling 2 as an A-QUALITY one
                self.modbusclient.writeSeedling3Quality(
                    int(seedlquality[2]))  # +1)) #lsmodb.QTY_A)  # Define seedling 3 as an A-QUALITY one
                self.modbusclient.cvFinishProcessing()  # Tell the PLC the processing has finished
            except:
                print(imgname)
                raise Exception("NAO DEU CERTINHO")
        # print("Thread OK")
        self.thread = Timer(0.2, self.check_plc)
        self.thread.start()
    """

    def on_connect_(self, userdata, flags, rc):
        print("Connected MQTT with result code " + str(rc))
        print("()()()()()()---PLANTINATOR AI PROCESSING UNIT---()()()()()()")
        # print(client + "/" + userdata + "/")
        # client.publish(topic_es,CMD_SLV_ON)
        # print("ON SENT")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        self.subscribe(self.topic4listenig)

    def on_message_(self, userdata, msg):
        self.topic = str(msg.topic)
        self.payload = str(msg.payload.decode('utf-8'))
        print("payload: " + self.payload)
        print("topic: " + self.topic)

        if self.topic == self.topic_parametros:
            # then parameters are going to be updated
            print("Updating parameters")
            self.updating_parameters(json.loads(self.payload))

    def updating_parameters(self, json_parameters_payload):
        self.g3min = json_parameters_payload["g3min"]
        self.g3max = json_parameters_payload["g3max"]
        self.g2min = json_parameters_payload["g2min"]
        self.g2max = json_parameters_payload["g2max"]
        self.g1min = json_parameters_payload["g1min"]
        self.g1max = json_parameters_payload["g1max"]
        self.hmin = json_parameters_payload["hmin"]
        self.hmax = json_parameters_payload["hmax"]
        self.score_threshold = json_parameters_payload["score_threshold"]
        self.left_threshold = json_parameters_payload["left_threshold"]
        self.right_threshold = json_parameters_payload["right_threshold"]

        print("Parameters updated:")
        print("g3min: ", self.g3min)
        print("g3max: ", self.g3max)
        print("g2min: ", self.g2min)
        print("g2max: ", self.g2max)
        print("g1min: ", self.g1min)
        print("g1max: ", self.g1max)
        print("hmin: ", self.hmin)
        print("hmax: ", self.hmax)
        print("score_threshold: ", self.score_threshold)
        print("left_threshold: ", self.left_threshold)
        print("right_threshold: ", self.right_threshold)

    def start_processing_image(self,img):
        print("Processing image")
        sn, sc = get_prediction(self.modelpath, img)
        return sn, sc