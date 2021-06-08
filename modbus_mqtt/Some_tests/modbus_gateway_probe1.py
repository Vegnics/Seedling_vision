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
import libseedlingmodbus as lsmodb
import struct

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

class modbusApp(tk.Tk):
    def __init__(self,tcpipdict):
        tk.Tk.__init__(self)
        self.tcpipdict=tcpipdict
        self.client = client(self.tcpipdict["modServerIp"], self.tcpipdict["modServerPort"]) #create de Modbus client
        self.client.connect()
        #self.publisher = registerPublisher(1.0,self.tcpipdict["brokerIp"],self.tcpipdict["brokerPort"],self.tcpipdict["modServerIp"],self.tcpipdict["modServerPort"])
        self.title("PRUEBAS MODBUS-TCP <--> MQTT")
        self.geometry("600x700")
        self.resizable(0,0)
        self.label_width = 35
        self.label1=tk.Label(self,text="Processed Trays",height=2,width=self.label_width,font= "Times 14 bold")
        self.label1.grid(row=0,column=0)
        self.registerTxt1=tk.Text(self,height=2,width=10)
        self.registerTxt1.grid(row=0,column=1)
        self.label2 = tk.Label(self, text="Classified Seedlings", height=2, width=self.label_width, font="Times 14 bold")
        self.label2.grid(row=1, column=0)
        self.registerTxt2 = tk.Text(self, height=2, width=10)
        self.registerTxt2.grid(row=1, column=1)

        self.label3 = tk.Label(self, text="Current Class A seedlings", height=2, width=self.label_width, font="Times 14 bold")
        self.label3.grid(row=2, column=0)
        self.registerTxt3 = tk.Text(self, height=2, width=10)
        self.registerTxt3.grid(row=2, column=1)

        self.label4 = tk.Label(self, text="Current Class B seedlings", height=2, width=self.label_width, font="Times 14 bold")
        self.label4.grid(row=3, column=0)
        self.registerTxt4 = tk.Text(self, height=2, width=10)
        self.registerTxt4.grid(row=3, column=1)

        self.label5 = tk.Label(self, text="Current Class C seedlings", height=2, width=self.label_width, font="Times 14 bold")
        self.label5.grid(row=4, column=0)
        self.registerTxt5 = tk.Text(self, height=2, width=10)
        self.registerTxt5.grid(row=4, column=1)

        self.label6 = tk.Label(self, text="Total Class A trays", height=2, width=self.label_width, font="Times 14 bold")
        self.label6.grid(row=5, column=0)
        self.registerTxt6 = tk.Text(self, height=2, width=10)
        self.registerTxt6.grid(row=5, column=1)

        self.label7 = tk.Label(self, text="Total Class B trays", height=2, width=self.label_width, font="Times 14 bold")
        self.label7.grid(row=6, column=0)
        self.registerTxt7 = tk.Text(self, height=2, width=10)
        self.registerTxt7.grid(row=6, column=1)

        self.label8 = tk.Label(self, text="Total Class C trays", height=2, width=self.label_width, font="Times 14 bold")
        self.label8.grid(row=7, column=0)
        self.registerTxt8 = tk.Text(self, height=2, width=10)
        self.registerTxt8.grid(row=7, column=1)

        self.label9 = tk.Label(self, text="X position", height=2, width=self.label_width, font="Times 14 bold")
        self.label9.grid(row=8, column=0)
        self.registerTxt9 = tk.Text(self, height=2, width=10)
        self.registerTxt9.grid(row=8, column=1)

        self.changeRegButton=tk.Button(self,text="Set Registers",command=self.RefreshReg)
        self.changeRegButton.grid(row=9,column=0)

    def RefreshReg(self):
        processed_Trays = int(self.registerTxt1.get("1.0","end"))
        self.client.write_register(4014,processed_Trays)

        classified_Seedlings = int(self.registerTxt2.get("1.0","end"))
        self.client.write_register(4015, classified_Seedlings)

        currentCASeedlings = int(self.registerTxt3.get("1.0","end"))
        self.client.write_register(4016, currentCASeedlings)

        currentCBSeedlings = int(self.registerTxt4.get("1.0","end"))
        self.client.write_register(4017, currentCBSeedlings)

        currentCCSeedlings = int(self.registerTxt5.get("1.0","end"))
        self.client.write_register(4018, currentCCSeedlings)

        totalCATrays = int(self.registerTxt6.get("1.0","end"))
        self.client.write_register(4019, totalCATrays)

        totalCBTrays = int(self.registerTxt7.get("1.0","end"))
        self.client.write_register(4020, totalCBTrays)

        totalCCTrays = int(self.registerTxt8.get("1.0","end"))
        self.client.write_register(4021, totalCCTrays)

        xPosition = float(self.registerTxt9.get("1.0","end"))
        bytesval = struct.pack('>f',xPosition)
        MSW = bytesval[0:2]
        LSW = bytesval[2:]
        MSval = struct.unpack('>H',MSW)
        LSval = struct.unpack('>H',LSW)
        print(MSval)
        self.client.write_register(4024,LSval[0])
        self.client.write_register(4023,MSval[0])


def thread_modbus_server():
    StartTcpServer(context, address=(tcpipvals["modServerIp"], tcpipvals["modServerPort"]))

myBlock=dataBlock(4015,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # seteamos dos registros
store = contextSlave(di=None, co=None, hr=myBlock, ir=None)
context =contextServer(slaves=store, single=True)

tcpipvals={"modServerIp":"192.168.0.17","modServerPort":5020,"brokerIp":"192.168.0.10","brokerPort":1884} #PUT HERE THE IP/PORT VALUES

modbus_thread = Thread(target=thread_modbus_server)
modbus_thread.start()

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

