from libseedlingmodbus import SeedlingModbusClient
import paho.mqtt.client as mqtt
from time import sleep


def modbus_holding_reg_decode(response):
    decoded_response = []
    bytes = response.encode()
    num_bytes = int(bytes[0])
    for i in range(int(num_bytes/2)):
        decoded_response.append(int.from_bytes(bytes[1+2*i:3+2*i],"big"))
    return decoded_response

def on_message(client, userdata, message):
    print("Received message '" + str(message.payload) + "' on topic '"
        + message.topic + "' with QoS " + str(message.qos))

tcpipvals={"modServerIp":"192.168.0.17","modServerPort":502,"brokerIp":"192.168.0.10","brokerPort":1884} #PUT HERE THE IP/PORT VALUES
client_1 = SeedlingModbusClient(tcpipvals["modServerIp"],tcpipvals["modServerPort"])
client_1.connect()
subscriber = mqtt.Client()
subscriber.connect(tcpipvals["brokerIp"],tcpipvals["brokerPort"],60)
subscriber.subscribe("robot/bandejas/alimentadora/bandejas")
subscriber.on_message = on_message
subscriber.loop_start()
#response = client_1.read_holding_registers(4014,7,unit=0x1)
#response = modbus_holding_reg_decode(response)
#print(response)

while(True):
    #response = client_1.read_holding_registers(4014, 8, unit=0x1)
    #response = modbus_holding_reg_decode(response)
    print("ModbusVal = {}".format(client_1.getProcessedTrays()))
    sleep(1.5)
