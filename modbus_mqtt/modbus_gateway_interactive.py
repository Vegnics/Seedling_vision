from pymodbus.server.sync import ModbusTcpServer,StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock as dataBlock
from pymodbus.datastore import ModbusSlaveContext as contextSlave
from pymodbus.datastore import ModbusServerContext as contextServer
from pymodbus.client.sync import ModbusTcpClient as client
from time import sleep
from threading import Timer,Thread
import tkinter as tk
import paho.mqtt.client as mqtt
from libseedlingmodbus import SeedlingModbusClient
import struct
from random import randint
import libseedlingmodbus as lsmodb
import sys

class registerPublisher():
    def __init__(self,t,brokerAdd,brokerPort,modbusAdd,modbusPort):
        self.t=t
        self.brokerAdd = brokerAdd
        self.brokerPort = brokerPort
        self.mqttc = mqtt.Client()
        self.mqttc.connect(self.brokerAdd, self.brokerPort, 60)
        self.modbusclient = SeedlingModbusClient(modbusAdd,modbusPort)
        self.thread = Timer(self.t,self.publishRegisters)
        self.thread.start()
    def publishRegisters(self):
        processed_Trays = self.modbusclient.getProcessedTrays()
        self.mqttc.publish("robot/bandejas/alimentadora/bandejas",str(processed_Trays))

        seedlingsForProcessing = self.modbusclient.getclassifiedSeedlings()
        self.mqttc.publish("robot/bandejas/alimentadora/porprocesar", str(seedlingsForProcessing))

        currentCASeedlings = self.modbusclient.getcurrentASeedlings()
        self.mqttc.publish("robot/bandejas/claseA/cantidad", str(currentCASeedlings))

        currentCBSeedlings = self.modbusclient.getcurrentBSeedlings()
        self.mqttc.publish("robot/bandejas/claseB/cantidad", str(currentCBSeedlings))

        currentCCSeedlings = self.modbusclient.getcurrentCSeedlings()
        self.mqttc.publish("robot/bandejas/claseC/cantidad", str(currentCCSeedlings))

        totalCATrays = self.modbusclient.gettotalATrays()
        self.mqttc.publish("robot/bandejas/claseA/bandejas", str(totalCATrays))

        totalCBTrays = self.modbusclient.gettotalBTrays()
        self.mqttc.publish("robot/bandejas/claseB/bandejas", str(totalCBTrays))

        totalCCTrays = self.modbusclient.gettotalCTrays()
        self.mqttc.publish("robot/bandejas/claseC/bandejas", str(totalCCTrays))

        self.thread = Timer(self.t, self.publishRegisters)
        self.thread.start()
    def start(self):
        self.thread.start()
    def cancel(self):
        self.thread.cancel()

class guiRefresher():
    def __init__(self,t,seedclient):
        self.t=t
        self.thread = Timer(self.t,self.getRegs)
        self.thread.start()
        self.seedclient = seedclient
        self.processedTrays = 0
        self.classifiedSeedlings = 0
        self.currentASeedlings = 0
    def getRegs(self):
        atrays = self.seedclient.getProcessedTrays()
    def start(self):
        self.thread.start()
    def cancel(self):
        self.thread.cancel()

class modbusApp(tk.Tk):
    def __init__(self,tcpipdict):
        tk.Tk.__init__(self)
        self.selectionq1 = tk.IntVar()
        self.selectionq2 = tk.IntVar()
        self.selectionq3 = tk.IntVar()
        self.qualityS1 = lsmodb.QTY_EMPTY
        self.qualityS2 = lsmodb.QTY_EMPTY
        self.qualityS3 = lsmodb.QTY_EMPTY
        self.tcpipdict=tcpipdict
        self.client = SeedlingModbusClient(self.tcpipdict["modServerIp"], self.tcpipdict["modServerPort"]) #create de Modbus client
        self.client.connect()
        self.readRegsthread = Timer(0.5,self.readRegs)
        self.readRegsthread.start()
        #self.publisher = registerPublisher(1.0,self.tcpipdict["brokerIp"],self.tcpipdict["brokerPort"],self.tcpipdict["modServerIp"],self.tcpipdict["modServerPort"])
        self.title("PLC <--> MODBUS-TCP<--> MQTT : INTERACTIVE MODE")
        self.geometry("1200x900")
        self.resizable(0,0)
        self.label_width = 25
        self.Label_processedTrays=tk.Label(self,text="Processed Trays",height=2,width=self.label_width,font= "Times 14 bold")
        self.Label_processedTrays.grid(row=0,column=0)
        self.Label_processedTraysnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_processedTraysnum.grid(row=0,column=1)

        self.Label_classifiedSeedlings = tk.Label(self, text="Classified Seedlings", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_classifiedSeedlings.grid(row=1, column=0)
        self.Label_classifiedSeedlingsnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_classifiedSeedlingsnum.grid(row=1, column=1)

        self.Label_currASeedlings = tk.Label(self, text="Current Class A seedlings", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_currASeedlings.grid(row=2, column=0)
        self.Label_currASeedlingsnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_currASeedlingsnum.grid(row=2, column=1)

        self.Label_currBSeedlings = tk.Label(self, text="Current Class B seedlings", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_currBSeedlings.grid(row=3, column=0)
        self.Label_currBSeedlingsnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_currBSeedlingsnum.grid(row=3, column=1)

        self.Label_currCSeedlings = tk.Label(self, text="Current Class C seedlings", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_currCSeedlings.grid(row=4, column=0)
        self.Label_currCSeedlingsnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_currCSeedlingsnum.grid(row=4, column=1)

        self.Label_totalAtrays = tk.Label(self, text="Total Class A trays", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_totalAtrays.grid(row=5, column=0)
        self.Label_totalAtraysnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_totalAtraysnum.grid(row=5, column=1)

        self.Label_totalBtrays = tk.Label(self, text="Total Class B trays", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_totalBtrays.grid(row=6, column=0)
        self.Label_totalBtraysnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_totalBtraysnum.grid(row=6, column=1)

        self.Label_totalCtrays = tk.Label(self, text="Total Class C trays", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_totalCtrays.grid(row=7, column=0)
        self.Label_totalCtraysnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_totalCtraysnum.grid(row=7, column=1)

        self.Label_XPosition = tk.Label(self, text="X position", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_XPosition.grid(row=8, column=0)
        self.Label_XPositionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold", bg="yellow")
        self.Label_XPositionnum.grid(row=8, column=1)

        self.Label_YPosition = tk.Label(self, text="Y position", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_YPosition.grid(row=9, column=0)
        self.Label_YPositionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_YPositionnum.grid(row=9, column=1)

        self.Label_Z1Position = tk.Label(self, text="Z1 position", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_Z1Position.grid(row=10, column=0)
        self.Label_Z1Positionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_Z1Positionnum.grid(row=10, column=1)

        self.Label_Z2Position = tk.Label(self, text="Z2 position", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_Z2Position.grid(row=11, column=0)
        self.Label_Z2Positionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_Z2Positionnum.grid(row=11, column=1)

        self.Label_Z3Position = tk.Label(self, text="Z3 position", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_Z3Position.grid(row=12, column=0)
        self.Label_Z3Positionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_Z3Positionnum.grid(row=12, column=1)

        self.Label_FTrayPosition = tk.Label(self, text="Feeder Tray Position", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_FTrayPosition.grid(row=0, column=2)
        self.Label_FTrayPositionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_FTrayPositionnum.grid(row=0, column=3)

        self.Label_ATrayPosition = tk.Label(self, text="Class A Tray Position", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_ATrayPosition.grid(row=1, column=2)
        self.Label_ATrayPositionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_ATrayPositionnum.grid(row=1, column=3)

        self.Label_BTrayPosition = tk.Label(self, text="Class B Tray Position", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_BTrayPosition.grid(row=2, column=2)
        self.Label_BTrayPositionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_BTrayPositionnum.grid(row=2, column=3)

        self.Label_CTrayPosition = tk.Label(self, text="Class C Tray Position", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_CTrayPosition.grid(row=3, column=2)
        self.Label_CTrayPositionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_CTrayPositionnum.grid(row=3, column=3)

        self.Label_NXPosition = tk.Label(self, text="Needles X Position", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_NXPosition.grid(row=4, column=2)
        self.Label_NXPositionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_NXPositionnum.grid(row=4, column=3)

        self.Label_NYPosition = tk.Label(self, text="Needles Y Position", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_NYPosition.grid(row=5, column=2)
        self.Label_NYPositionnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_NYPositionnum.grid(row=5, column=3)

        self.Label_GripperStat = tk.Label(self, text="Gripper status", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_GripperStat.grid(row=6, column=2)
        self.Label_GripperStatnum = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_GripperStatnum.grid(row=6, column=3)

        self.Label_Alarm = tk.Label(self, text="Alarm", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_Alarm.grid(row=7, column=2)
        self.Label_Alarmval = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_Alarmval.grid(row=7, column=3)

        self.Label_PlcInstruct = tk.Label(self, text="PLC instruction", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_PlcInstruct.grid(row=8, column=2)
        self.Label_PlcInstructval = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="yellow")
        self.Label_PlcInstructval.grid(row=8, column=3)

        self.Label_CvStatus = tk.Label(self, text="CV status", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_CvStatus.grid(row=9, column=2)
        self.Label_CvStatusval = tk.Label(self, text="0", height=2, width=self.label_width,font="Times 14 bold",bg="orange")
        self.Label_CvStatusval.grid(row=9, column=3)

        self.Label_S1Quality = tk.Label(self, text="Seedling 1 Quality", height=2, width=self.label_width, font="Times 14 bold")
        self.Label_S1Quality.grid(row=10, column=2)
        self.Label_S1Qualityval = tk.Label(self, text="0", height=2, width=self.label_width, font="Times 14 bold",bg="orange")
        self.Label_S1Qualityval.grid(row=10, column=3)

        self.Label_S2Quality = tk.Label(self, text="Seedling 2 Quality", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_S2Quality.grid(row=11, column=2)
        self.Label_S2Qualityval = tk.Label(self, text="0", height=2, width=self.label_width,font="Times 14 bold",bg="orange")
        self.Label_S2Qualityval.grid(row=11, column=3)

        self.Label_S3Quality = tk.Label(self, text="Seedling 3 Quality", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_S3Quality.grid(row=12, column=2)
        self.Label_S3Qualityval = tk.Label(self, text="0", height=2, width=self.label_width,font="Times 14 bold",bg="orange")
        self.Label_S3Qualityval.grid(row=12, column=3)

        self.Frame_S1Quality = tk.Frame(self,height=2,width=self.label_width)
        self.Frame_S1Quality.grid(row=1,column=4)

        self.Frame_S2Quality = tk.Frame(self, height=2, width=self.label_width)
        self.Frame_S2Quality.grid(row=3, column=4)

        self.Frame_S3Quality = tk.Frame(self, height=2, width=self.label_width)
        self.Frame_S3Quality.grid(row=5, column=4)

        self.Label_S1SelQuality = tk.Label(self, text="Seedling 1 Quality", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_S1SelQuality.grid(row=0, column=4)
        self.RButton_S1_NS = tk.Radiobutton(self.Frame_S1Quality, text="Empty", variable=self.selectionq1,value=lsmodb.QTY_EMPTY, command=self.selq1)
        self.RButton_S1_A = tk.Radiobutton(self.Frame_S1Quality,text="A",variable=self.selectionq1,value=lsmodb.QTY_A,command=self.selq1)
        self.RButton_S1_B = tk.Radiobutton(self.Frame_S1Quality, text="B", variable=self.selectionq1, value=lsmodb.QTY_B,command=self.selq1)
        self.RButton_S1_C = tk.Radiobutton(self.Frame_S1Quality, text="C", variable=self.selectionq1, value=lsmodb.QTY_C,command=self.selq1)
        self.RButton_S1_NS.grid(row=0,column=0)
        self.RButton_S1_A.grid(row=0,column=1)
        self.RButton_S1_B.grid(row=0,column=2)
        self.RButton_S1_C.grid(row=0,column=3)

        self.Label_S2SelQuality = tk.Label(self, text="Seedling 2 Quality", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_S2SelQuality.grid(row=2, column=4)
        self.RButton_S2_NS = tk.Radiobutton(self.Frame_S2Quality, text="Empty", variable=self.selectionq2,value=lsmodb.QTY_EMPTY, command=self.selq2)
        self.RButton_S2_A = tk.Radiobutton(self.Frame_S2Quality, text="A", variable=self.selectionq2,value=lsmodb.QTY_A, command=self.selq2)
        self.RButton_S2_B = tk.Radiobutton(self.Frame_S2Quality, text="B", variable=self.selectionq2,value=lsmodb.QTY_B, command=self.selq2)
        self.RButton_S2_C = tk.Radiobutton(self.Frame_S2Quality, text="C", variable=self.selectionq2,value=lsmodb.QTY_C, command=self.selq2)
        self.RButton_S2_NS.grid(row=0, column=0)
        self.RButton_S2_A.grid(row=0, column=1)
        self.RButton_S2_B.grid(row=0, column=2)
        self.RButton_S2_C.grid(row=0, column=3)

        self.Label_S3SelQuality = tk.Label(self, text="Seedling 3 Quality", height=2, width=self.label_width,font="Times 14 bold")
        self.Label_S3SelQuality.grid(row=4, column=4)
        self.RButton_S3_NS = tk.Radiobutton(self.Frame_S3Quality, text="Empty", variable=self.selectionq3,value=lsmodb.QTY_EMPTY, command=self.selq3)
        self.RButton_S3_A = tk.Radiobutton(self.Frame_S3Quality, text="A", variable=self.selectionq3,value=lsmodb.QTY_A, command=self.selq3)
        self.RButton_S3_B = tk.Radiobutton(self.Frame_S3Quality, text="B", variable=self.selectionq3,value=lsmodb.QTY_B, command=self.selq3)
        self.RButton_S3_C = tk.Radiobutton(self.Frame_S3Quality, text="C", variable=self.selectionq3,value=lsmodb.QTY_C, command=self.selq3)
        self.RButton_S3_NS.grid(row=0, column=0)
        self.RButton_S3_A.grid(row=0, column=1)
        self.RButton_S3_B.grid(row=0, column=2)
        self.RButton_S3_C.grid(row=0, column=3)

        self.setButton=tk.Button(self,text="Set Quality Values",command=self.setQuality,bg="green")
        self.setButton.grid(row=6,column=4)


    def readRegs(self):
        self.Label_processedTraysnum.config(text="{}".format(self.client.getProcessedTrays()))
        self.Label_classifiedSeedlingsnum.config(text="{}".format(self.client.getclassifiedSeedlings()))
        self.Label_currASeedlingsnum.config(text="{}".format(self.client.getcurrentASeedlings()))
        self.Label_currBSeedlingsnum.config(text="{}".format(self.client.getcurrentBSeedlings()))
        self.Label_currCSeedlingsnum.config(text="{}".format(self.client.getcurrentCSeedlings()))
        self.Label_totalAtraysnum.config(text="{}".format(self.client.gettotalATrays()))
        self.Label_totalBtraysnum.config(text="{}".format(self.client.gettotalBTrays()))
        self.Label_totalCtraysnum.config(text="{}".format(self.client.gettotalCTrays()))
        self.Label_XPositionnum.config(text="{:3.2f}".format(self.client.getXPosition()))
        self.Label_YPositionnum.config(text="{:3.2f}".format(self.client.getYPosition()))
        self.Label_Z1Positionnum.config(text="{:3.2f}".format(self.client.getZ1Position()))
        self.Label_Z2Positionnum.config(text="{:3.2f}".format(self.client.getZ2Position()))
        self.Label_Z3Positionnum.config(text="{:3.2f}".format(self.client.getZ3Position()))
        self.Label_FTrayPositionnum.config(text="{:3.2f}".format(self.client.getFeederTrayPosition()))
        self.Label_ATrayPositionnum.config(text="{:3.2f}".format(self.client.getClassATrayPosition()))
        self.Label_BTrayPositionnum.config(text="{:3.2f}".format(self.client.getClassBTrayPosition()))
        self.Label_CTrayPositionnum.config(text="{:3.2f}".format(self.client.getClassCTrayPosition()))
        self.Label_NXPositionnum.config(text="{:3.2f}".format(self.client.getNeedlesXPosition()))
        self.Label_NYPositionnum.config(text="{:3.2f}".format(self.client.getNeedlesYPosition()))
        self.Label_GripperStatnum.config(text="{}".format(self.client.getGripperStatus()))
        self.Label_Alarmval.config(text="{}".format(self.client.getAlarms()))
        #self.Label_PlcInstructval.config(text="{}".format(self.client.getPLCInstruction()))
        #print("thread working")
        self.checkPLCInstruction()
        self.checkCvSystem()
        self.checkSeedlings()
        self.readRegsthread = Timer(0.5, self.readRegs)
        self.readRegsthread.start()

    def selq1(self):
        self.qualityS1 = self.selectionq1.get()

    def selq2(self):
        self.qualityS2 = self.selectionq2.get()

    def selq3(self):
        self.qualityS3 = self.selectionq3.get()

    def setQuality(self):
        #cvStatus = int(self.cvStatTxt.get("1.0","end"))
        #self.client.writeCvStatus(cvStatus)
        #s1quality = int(self.S1QualityTxt.get("1.0","end"))
        self.client.writeSeedling1Quality(self.qualityS1)

        #s2quality = int(self.S2QualityTxt.get("1.0", "end"))
        self.client.writeSeedling2Quality(self.qualityS2)

        #s3quality = int(self.S3QualityTxt.get("1.0", "end"))
        self.client.writeSeedling3Quality(self.qualityS3)
        sleep(0.3)
        self.client.writeCvStatus(lsmodb.CV_PROCFINISHED_STAT)
        """
        xPosition = float(self.registerTxt9.get("1.0","end"))
        bytesval = struct.pack('>f',xPosition)
        MSW = bytesval[0:2]
        LSW = bytesval[2:]
        MSval = struct.unpack('>H',MSW)
        LSval = struct.unpack('>H',LSW)
        print(MSval)
        self.client.write_register(4024,LSval[0])
        self.client.write_register(4023,MSval[0])
        """

    def checkPLCInstruction(self):
        plcInstruction = self.client.getPLCInstruction()
        if plcInstruction == lsmodb.PLC_NOORDER_INST:
            self.Label_PlcInstructval.config(text="No order")
        elif plcInstruction == lsmodb.PLC_PROCODD_INST:
            self.Label_PlcInstructval.config(text="Odd seedlings")
            #self.client.writeCvStatus(lsmodb.CV_PROCESSING_STAT)
            #sleep(2)
            #self.client.writeSeedling1Quality(randint(0,1))
            #sleep(0.1)
            #self.client.writeSeedling2Quality(randint(0,1))
            #sleep(0.1)
            #self.client.writeSeedling3Quality(randint(0,1))
            self.client.writeCvStatus(lsmodb.CV_PROCESSING_STAT)

        elif plcInstruction == lsmodb.PLC_PROCEVEN_INST:
            self.Label_PlcInstructval.config(text="Even seedlings")
            #self.client.writeCvStatus(lsmodb.CV_PROCESSING_STAT)
            #sleep(2)
            #self.client.writeSeedling1Quality(randint(0, 1))
            #sleep(0.1)
            #self.client.writeSeedling2Quality(randint(0, 1))
            #sleep(0.1)
            #self.client.writeSeedling3Quality(randint(0, 1))
            self.client.writeCvStatus(lsmodb.CV_PROCESSING_STAT)

        elif plcInstruction == lsmodb.PLC_ACK_INST:
            self.Label_PlcInstructval.config(text="Acknowledged")
            sleep(0.5)
            self.client.writeCvStatus(lsmodb.CV_WAITING_STAT)
            #self.client.writeSeedling1Quality(0)
            #self.client.writeSeedling2Quality(0)
            #self.client.writeSeedling3Quality(0)

    def checkCvSystem(self):
        cvStatus = self.client.getCvStatus()  # GET Computer vision system status
        if cvStatus == lsmodb.CV_WAITING_STAT:
            self.Label_CvStatusval.config(text="Waiting...")
        elif cvStatus == lsmodb.CV_PROCESSING_STAT:
            self.Label_CvStatusval.config(text="Processing")
        elif cvStatus == lsmodb.CV_CAMERROR_STAT:
            self.Label_CvStatusval.config(text="Camera error")
        elif cvStatus == lsmodb.CV_PROCERROR_STAT:
            self.Label_CvStatusval.config(text="Processing error")
        elif cvStatus == lsmodb.CV_PROCFINISHED_STAT:
            self.Label_CvStatusval.config(text="Processing Finished")

    def checkSeedlings(self):
        s1Quality = self.client.getSeedling1Quality()
        if s1Quality == lsmodb.QTY_EMPTY:
            self.Label_S1Qualityval.config(text="Empty")
        elif s1Quality == lsmodb.QTY_A:
            self.Label_S1Qualityval.config(text="A")
        elif s1Quality == lsmodb.QTY_B:
            self.Label_S1Qualityval.config(text="B")
        elif s1Quality == lsmodb.QTY_C:
            self.Label_S1Qualityval.config(text="C")

        s2Quality = self.client.getSeedling2Quality()
        if s2Quality == lsmodb.QTY_EMPTY:
            self.Label_S2Qualityval.config(text="Empty")
        elif s2Quality == lsmodb.QTY_A:
            self.Label_S2Qualityval.config(text="A")
        elif s2Quality == lsmodb.QTY_B:
            self.Label_S2Qualityval.config(text="B")
        elif s2Quality == lsmodb.QTY_C:
            self.Label_S2Qualityval.config(text="C")

        s3Quality = self.client.getSeedling3Quality()
        if s3Quality == lsmodb.QTY_EMPTY:
            self.Label_S3Qualityval.config(text="Empty")
        elif s3Quality == lsmodb.QTY_A:
            self.Label_S3Qualityval.config(text="A")
        elif s3Quality == lsmodb.QTY_B:
            self.Label_S3Qualityval.config(text="B")
        elif s3Quality == lsmodb.QTY_C:
            self.Label_S3Qualityval.config(text="C")

def thread_modbus_server():
    StartTcpServer(context, address=(tcpipvals["modServerIp"], tcpipvals["modServerPort"]))


args = sys.argv

modServerIp = "192.168.1.103"
modServerPort = 502
brokerIp = "192.168.0.10"
brokerPort = 1884

if "-serverIp" in args:
    idx = args.index("-serverIp")
    try:
        modServerIp = args[idx+1]
    except:
        raise Exception("Server IP wasn't specified")

if "-serverPort" in args:
    idx = args.index("-serverPort")
    try:
        modServerPort = int(args[idx+1])
    except:
        raise Exception("Server Port wasn't specified or is not valid" )

if "-brokerIp" in args:
    idx = args.index("-brokerIp")
    try:
        brokerIp = args[idx + 1]
    except:
        raise Exception("Broker Ip wasn't specified")

if "-brokerPort" in args:
    idx = args.index("-brokerPort")
    try:
        brokerPort = int(args[idx + 1])
    except:
        raise Exception("Broker Port wasn't specified or is not valid")

_dataBlockSize = 91

myBlock=dataBlock(4015,[0]*_dataBlockSize)# seteamos dos registros
myBlock2 = dataBlock(4102,[0,0,0])

store = contextSlave(di=None, co=None, hr=myBlock, ir=None)

context =contextServer(slaves=store, single=True)

tcpipvals={"modServerIp":modServerIp,"modServerPort":modServerPort,"brokerIp":brokerIp,"brokerPort":brokerPort} #PUT HERE THE IP/PORT VALUES

modbus_thread = Thread(target=thread_modbus_server)
modbus_thread.start()
print("Modbus Server Started ...")

sleep(2)

#client_1 = client("192.168.2.105",5020)
#client_1.connect()
try:
    mainApp = modbusApp(tcpipvals)
    mainApp.mainloop()

except KeyboardInterrupt:
    print("exiting ...")

#for i in range(250):
#    client_1.write_register(4014,i)
#    sleep(1.5)