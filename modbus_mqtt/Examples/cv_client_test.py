from libseedlingmodbus import SeedlingModbusClient
import libseedlingmodbus as lsmodb
import sys
from time import time,sleep

args = sys.argv

modServerIp = "192.168.1.103"
modServerPort = 502

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
sleep(2)
cv_Client.writeCvStatus(lsmodb.CV_PROCESSING_STAT)
sleep(2)
cv_Client.cvFinishProcessing()
