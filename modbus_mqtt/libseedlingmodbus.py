import pymodbus
#!/usr/bin/python3.6
"""
BLOCKS:
di: Discrete Inputs
co: Coil Outputs
hr: Holding Registers
ir: Input Registers
"""

from pymodbus.client.sync import ModbusTcpClient as client
from time import sleep,time
from threading import Timer,Thread
import struct
import numpy as np

#CV_NO_CONNECTED = 0 -> This should be implemented on the PLC first
CV_WAITING_STAT = 0
CV_PROCESSING_STAT = 1
CV_CAMERROR_STAT = 2
CV_PROCERROR_STAT = 3
CV_PROCFINISHED_STAT = 4

PLC_NOORDER_INST = 0
PLC_PROCODD_INST = 1
PLC_PROCEVEN_INST = 2
PLC_ACK_INST = 3

QTY_EMPTY = 0
QTY_A = 1
QTY_B = 2
QTY_C = 3

class SeedlingModbusClient(client):
    def __init__(self,ServerIp,ServerPort,usePlcTimeout=True,plcTimeout = 5):
        client.__init__(self,ServerIp,ServerPort)
        self.serverIp = ServerIp
        self.serverPort = ServerPort
        self.plcTimeout = plcTimeout
        self.usePlcTimeout = usePlcTimeout
        self.__timingFlag = False
        self.__timeoutFlag = False
        self.regCodes = {"processedTrays":4014,
                         "classifiedSeedlings":4015,
                         "currentASeedlings":4016,
                         "currentBSeedlings":4017,
                         "currentCSeedlings":4018,
                         "totalATrays":4019,
                         "totalBTrays":4020,
                         "totalCTrays":4021,
                         "currentXPos":4022,
                         "currentYPos":4024,
                         "currentZ1Pos":4026,
                         "currentZ2Pos":4028,
                         "currentZ3Pos":4030,
                         "currentFeederTrayPos":4032,
                         "currentATrayPos":4034,
                         "currentBTrayPos":4036,
                         "currentCTrayPos":4038,
                         "currentNeedlesX":4040,
                         "currentNeedlesZ":4042,
                         "gripperStatus":4044,
                         "alarms":4045,
                         "plcInstruction":4046,
                         "detectedSeedlings": 4047,
                         "cvStatus":4101,
                         "seedling1Quality":4102,
                         "seedling2Quality":4103,
                         "seedling3Quality":4104,
                         "z1Correction": 4105,
                         "z2Correction": 4107,
                         "z3Correction": 4109
                         }
    def connectToServer(self):
        return self.connect()

    def __checkPLCTimeOut(self):
        self.__timeoutThread = Timer(self.plcTimeout, self.__checkPLCTimeOut)
        if self.__timingFlag is False:
            self.__timeoutThread.start()
            self.__timingFlag = True
        else:
            self.__timingFlag = False
            self.__timeoutFlag = True

    def readModbusHoldReg(self,regNum):
        response = self.read_holding_registers(regNum,1,unit=0x1)
        rep_bytes = response.encode()
        decoded_response = int.from_bytes(rep_bytes[1:], "big")
        return decoded_response

    def readModbusReal(self,regNum):
        response = self.read_holding_registers(regNum, 1, unit=0x1)
        rep_bytes = response.encode()
        response2 = self.response = self.read_holding_registers(regNum+1, 1, unit=0x1)
        rep_bytes2 = response2.encode()
        MSval = int.from_bytes(rep_bytes2[1:],"big")
        LSval = int.from_bytes(rep_bytes[1:],"big")
        TOTval = MSval*(256**2) + LSval
        TOTbytes = TOTval.to_bytes(4,"big",signed=False)
        val = struct.unpack('>f',TOTbytes)
        return val
    def writeModbusReal(self,regNum,val):
        _val = np.float32(val)
        _bytes_int = struct.pack('>f',_val)
        MSVAL = 256*_bytes_int[0]+_bytes_int[1]
        LSVAL = 256* _bytes_int[2] + _bytes_int[3]
        self.write_register(regNum+1, MSVAL)
        self.write_register(regNum, LSVAL)

    def getProcessedTrays(self):
        return self.readModbusHoldReg(self.regCodes["processedTrays"])

    def getclassifiedSeedlings(self):
        return self.readModbusHoldReg(self.regCodes["classifiedSeedlings"])

    def getcurrentASeedlings(self):
        return self.readModbusHoldReg(self.regCodes["currentASeedlings"])

    def getcurrentBSeedlings(self):
        return self.readModbusHoldReg(self.regCodes["currentBSeedlings"])

    def getcurrentCSeedlings(self):
        return self.readModbusHoldReg(self.regCodes["currentCSeedlings"])

    def gettotalATrays(self):
        return self.readModbusHoldReg(self.regCodes["totalATrays"])

    def gettotalBTrays(self):
        return self.readModbusHoldReg(self.regCodes["totalBTrays"])

    def gettotalCTrays(self):
        return self.readModbusHoldReg(self.regCodes["totalCTrays"])

    def getXPosition(self):
        xval = self.readModbusReal(self.regCodes["currentXPos"])
        return xval[0]

    def getYPosition(self):
        yval = self.readModbusReal(self.regCodes["currentYPos"])
        return yval[0]

    def getZ1Position(self):
        z1val = self.readModbusReal(self.regCodes["currentZ1Pos"])
        return z1val[0]

    def getZ2Position(self):
        z2val = self.readModbusReal(self.regCodes["currentZ2Pos"])
        return z2val[0]

    def getZ3Position(self):
        z3val = self.readModbusReal(self.regCodes["currentZ3Pos"])
        return z3val[0]

    def getFeederTrayPosition(self):
        ftval = self.readModbusReal(self.regCodes["currentFeederTrayPos"])
        return ftval[0]

    def getClassATrayPosition(self):
        catval = self.readModbusReal(self.regCodes["currentATrayPos"])
        return catval[0]

    def getClassBTrayPosition(self):
        cbtval = self.readModbusReal(self.regCodes["currentBTrayPos"])
        return cbtval[0]

    def getClassCTrayPosition(self):
        cctval = self.readModbusReal(self.regCodes["currentCTrayPos"])
        return cctval[0]

    def getNeedlesXPosition(self):
        nxval = self.readModbusReal(self.regCodes["currentNeedlesX"])
        return nxval[0]

    def getNeedlesZPosition(self):
        nyval = self.readModbusReal(self.regCodes["currentNeedlesZ"])
        return nyval[0]

    def getGripperStatus(self):
        return self.readModbusHoldReg(self.regCodes["gripperStatus"])

    def getAlarms(self):
        return self.readModbusHoldReg(self.regCodes["alarms"])

    def getPLCInstruction(self):
        return self.readModbusHoldReg(self.regCodes["plcInstruction"])

    def writeCvStatus(self,val):
        self.write_register(self.regCodes["cvStatus"],val)

    def writeSeedling1Quality(self,val):
        self.write_register(self.regCodes["seedling1Quality"],val)

    def writeSeedling2Quality(self,val):
        self.write_register(self.regCodes["seedling2Quality"],val)

    def writeSeedling3Quality(self,val):
        self.write_register(self.regCodes["seedling3Quality"],val)

    def getCvStatus(self):
        return self.readModbusHoldReg(self.regCodes["cvStatus"])

    def getSeedling1Quality(self):
        return self.readModbusHoldReg(self.regCodes["seedling1Quality"])

    def getSeedling2Quality(self):
        return self.readModbusHoldReg(self.regCodes["seedling2Quality"])

    def getSeedling3Quality(self):
        return self.readModbusHoldReg(self.regCodes["seedling3Quality"])

    def cvFinishProcessing(self):
        self.writeCvStatus(CV_PROCFINISHED_STAT)
        sleep(0.4)
        if self.usePlcTimeout is True:
            self.__checkPLCTimeOut()
            while ((self.getPLCInstruction() != PLC_ACK_INST) and  (self.__timeoutFlag is False)):
                pass
            if self.__timeoutFlag is True:
                print("WARNING: Acknowledged message didn't arrive within {} seconds".format(self.plcTimeout))
            self.__timeoutFlag = False
            self.writeCvStatus(CV_WAITING_STAT)
        else:
            while (self.getPLCInstruction() != PLC_ACK_INST):
                pass
            self.writeCvStatus(CV_WAITING_STAT)

    def writeZcorrection(self,z1val,z2val,z3val):
        self.writeModbusReal(self.regCodes["z1Correction"],z1val)
        self.writeModbusReal(self.regCodes["z2Correction"],z2val)
        self.writeModbusReal(self.regCodes["z3Correction"],z3val)

    def getZcorrection(self):
        Z1_corr = self.readModbusReal(self.regCodes["z1Correction"])
        Z2_corr = self.readModbusReal(self.regCodes["z2Correction"])
        Z3_corr = self.readModbusReal(self.regCodes["z3Correction"])
        return Z1_corr[0],Z2_corr[0],Z3_corr[0]







