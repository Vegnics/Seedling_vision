from pymodbus.client.sync import ModbusTcpClient as client
from time import sleep
from libseedlingmodbus import SeedlingModbusClient


def modbus_holding_reg_decode(response):
    decoded_response = []
    bytes = response.encode()
    num_bytes = int(bytes[0])
    for i in range(int(num_bytes/2)):
        decoded_response.append(int.from_bytes(bytes[1+2*i:3+2*i],"big"))
    return decoded_response

_client = {"ip":"192.168.1.103", "port":502}

client1 = SeedlingModbusClient(_client["ip"],_client["port"])
client1.connectToServer()

#response = client_1.read_holding_registers(4014,7,unit=0x1)
#response = modbus_holding_reg_decode(response)
#print(response)

while(True):
    #response = client_1.read_holding_registers(4014, 8, unit=0x1)
    #response = modbus_holding_reg_decode(response)
    #response = client1.getProcessedTrays()
    #response = client1.getcurrentASeedlings()
    response = client1.readModbusHoldReg(4104)
    print(response)
    #client1.writeStatus(16)
    #client1.write_register(4, 65)
    sleep(1.5)
