from libseedlingmodbus import SeedlingModbusClient
import libseedlingmodbus as lsmodb
import sys
from time import time,sleep

args = sys.argv

modServerIp = "192.168.0.17"
modServerPort = 5020

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

cv_Client = SeedlingModbusClient(modServerIp,modServerPort)

while True:
    if cv_Client.getPLCInstruction() == lsmodb.PLC_PROCEVEN_INST:
        #Process even seedlings
        cv_Client.writeCvStatus(lsmodb.CV_PROCESSING_STAT) # Tell the PLC you've received the instruction and you're processing.
        cv_Client.writeSeedling1Quality(lsmodb.QTY_A) # Define seedling 1 as an A-QUALITY one
        cv_Client.writeSeedling2Quality(lsmodb.QTY_A)  # Define seedling 2 as an A-QUALITY one
        cv_Client.writeSeedling3Quality(lsmodb.QTY_A)  # Define seedling 3 as an A-QUALITY one
        cv_Client.cvFinishProcessing() #Tell the PLC the processing has finished
    elif cv_Client.getPLCInstruction() == lsmodb.PLC_PROCODD_INST:
        #Process odd seedlings
        cv_Client.writeCvStatus(lsmodb.CV_PROCESSING_STAT)  # Tell the PLC you've received the instruction and you're processing.
        cv_Client.writeSeedling1Quality(lsmodb.QTY_A)  # Define seedling 1 as an A-QUALITY one
        cv_Client.writeSeedling2Quality(lsmodb.QTY_A)  # Define seedling 2 as an A-QUALITY one
        cv_Client.writeSeedling3Quality(lsmodb.QTY_A)  # Define seedling 3 as an A-QUALITY one
        cv_Client.cvFinishProcessing() #Tell the PLC the processing has finished

