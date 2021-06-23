from pymodbus.server.sync import ModbusTcpServer,StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock as dataBlock
from pymodbus.datastore import ModbusSlaveContext as contextSlave
from pymodbus.datastore import ModbusServerContext as contextServer
from pymodbus.client.sync import ModbusTcpClient as client
from time import sleep
from threading import Timer,Thread

def thread_modbus_server():
    StartTcpServer(context, address=(tcpipvals["serverIp"], tcpipvals["serverPort"]))

myBlock=dataBlock(4015,[25,67,65000]) # seteamos dos registros
store = contextSlave(di=None, co=None, hr=myBlock, ir=None)
context =contextServer(slaves=store, single=True)

tcpipvals={"serverIp":"192.168.2.105","serverPort":5020}
#StartTcpServer(context, address=(tcpipvals["serverIp"],tcpipvals["serverPort"]))

modbus_thread = Thread(target=thread_modbus_server)
modbus_thread.start()

sleep(2)

client_1 = client("192.168.2.105",5020)
client_1.connect()

for i in range(250):
    client_1.write_register(4014,i)
    sleep(1.5)


