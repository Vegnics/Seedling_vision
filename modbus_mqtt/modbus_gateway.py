from pymodbus.server.sync import ModbusTcpServer,StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock as dataBlock
from pymodbus.datastore import ModbusSlaveContext as contextSlave
from pymodbus.datastore import ModbusServerContext as contextServer
from pymodbus.client.sync import ModbusTcpClient as client
from time import sleep
import paho.mqtt.client as mqtt
from threading import Timer,Thread
from libseedlingmodbus import SeedlingModbusClient
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

_dataBlockSize = 97

myBlock=dataBlock(4015,[0]*_dataBlockSize)# initialize all the registers to 0

store = contextSlave(di=None, co=None, hr=myBlock, ir=None)

context =contextServer(slaves=store, single=True)

tcpipvals={"modServerIp":modServerIp,"modServerPort":modServerPort,"brokerIp":brokerIp,"brokerPort":brokerPort} #PUT HERE THE IP/PORT VALUES

modbus_thread = Thread(target=thread_modbus_server)
modbus_thread.start()
print("Modbus Server Started ...")
while True:
    sleep(3)
    pass