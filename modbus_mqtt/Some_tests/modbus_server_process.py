import pymodbus
#!/usr/bin/python3.6
"""
EJEMPLO DE COMUNICACIÓN MODBUSTCP BIDIRECCIONAL
Pymodbus server on port 5020
Pydmodbus client on port 5021

BLOQUES:
di: Discrete Inputs
co: Coil Outputs
hr: Holding Registers
ir: Input Registers
"""

import tkinter as tk
#from pymodbus.client.sync import ModbusTcpClient
from pymodbus.server.sync import ModbusTcpServer,StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock as dataBlock
from pymodbus.datastore import ModbusSlaveContext as contextSlave
from pymodbus.datastore import ModbusServerContext as contextServer
from time import sleep
from threading import Timer,Thread

class readHoldRegisters():
    def __init__(self,t,client,address,numReg):
        self.t=t
        self.client=client
        self.address=address
        self.numReg=numReg
        self.thread = Timer(self.t,self.printregisters)
        self.log=" "
    def printregisters(self):
        try:
            response=self.client.read_holding_registers(self.address,self.numReg,unit=0x1)
            print(response.registers)
            self.log="Server OK"
        except:
            self.log="Server not available"
        self.thread = Timer(self.t, self.printregisters)
        self.thread.start()
    def start(self):
        self.thread.start()
    def cancel(self):
        self.thread.cancel()

class refreshRegisters():
    def __init__(self,context,tcpdict):
        self.context=context
        self.tcpipdict=tcpdict
        self.thread = Timer(0.3,self.refresh)
    def refresh(self):
        StartTcpServer(self.context, address=(self.tcpipdict["serverIp"],self.tcpipdict["serverPort"]))
        self.thread = Timer(0.3, self.refresh)
        self.thread.start()
    def start(self):
        self.thread.start()
    def cancel(self):
        self.thread.cancel()


class modbusApp(tk.Tk):
    def __init__(self,tcpipdict):
        tk.Tk.__init__(self)
        self.tcpipdict=tcpipdict
        self.threadreg = Timer(0.3, self.refresh)
        self.regsval=[0,0]
        self.myBlock=dataBlock(4015,self.regsval)
        self.store = contextSlave(di=None, co=None, hr=self.myBlock, ir=None)
        self.context =contextServer(slaves=self.store, single=True)
        self.serverthread=refreshRegisters(self.context,self.tcpipdict)
        self.serverthread.start()
        self.title("PRUEBAS MODBUS-TCP")
        self.geometry("600x600")
        self.resizable(0,0)
        self.label1=tk.Label(self,text="Registro 1",height=2,width=10,font= "Times 14 bold")
        self.label1.grid(row=0,column=0)
        self.register1Main = tk.Label(self)
        self.register1Main.grid(row=1,column=0)
        self.registerTxt1=tk.Text(self.register1Main,height=2,width=10)
        self.registerTxt1.grid(row=0,column=0)
        self.registerLab1=tk.Label(self.register1Main,bg="PaleGreen1",bd=2,highlightthickness=2,height=2,width=10)
        self.registerLab1.grid(row=0,column=1)
        self.changeReg1Button=tk.Button(self.register1Main,text="Setear",command=self.reg1Button)
        self.changeReg1Button.grid(row=0,column=2)

        self.label2=tk.Label(self,text="Registro 2",height=2,width=10,font= "Times 14 bold")
        self.label2.grid(row=2,column=0)
        self.register2Main = tk.Label(self)
        self.register2Main.grid(row=3,column=0)
        self.registerTxt2=tk.Text(self.register2Main,height=2,width=10)
        self.registerTxt2.grid(row=0,column=0)
        self.registerLab2=tk.Label(self.register2Main,bg="PaleGreen1",bd=2,highlightthickness=2,height=2,width=10)
        self.registerLab2.grid(row=0,column=1)
        self.changeReg2Button=tk.Button(self.register2Main,text="Setear",command=self.reg2Button)
        self.changeReg2Button.grid(row=0,column=2)

        self.labelDebbug=tk.Label(self,height=4,width=35,bg="orange2",anchor="nw",text="LOGS:")
        self.labelDebbug.grid(row=4,column=0,padx=10,pady=15)
        self.labelDebbug.configure(text="LOGS:"+"\n\r"+self.readReg.log)

    def reg2Button(self):
        val=int(self.registerTxt2.get("1.0","end"))
        self.registerLab2.configure(text=str(val))
        self.myBlock = dataServer(4015, [self.regsval[0],val])

    def reg1Button(self):
        val=int(self.registerTxt1.get("1.0","end"))
        self.registerLab1.configure(text=str(val))
        self.myBlock = dataServer(4015, [val,self.regsval[1]])


"""
#Asignamos valores para conexión TCP/IP
serverIp="192.168.1.107" #La IP de esta PC
serverPort=5020          #El puerto para la escucha
clientIp="192.168.1.100" #La IP de la otra PC
clientPort=5021          #El puerto para la lectura/escritura
"""


tcpipvals={"serverIp":"192.168.1.102","serverPort":5020,"clientIp":"192.168.1.102","clientPort":5021}

mainapp=modbusApp(tcpipvals)
mainapp.mainloop()

"""
#CONFIGURAMOS EL SERVIDOR MODBUS TCP
myBlock=dataServer(4015,[1000,8000])                  #Creamos el bloque de datos del servidor Modbus TCP
store=contextSlave(di=None,co=None,hr=myBlock,ir=None)#Asignamos los bloques de datos al servidor Modbus TCP
context=contextServer(slaves=store,single=True)
StartTcpServer(context,address=(serverIp,serverPort)) #Encendemos el servidor ModbusTCP
"""
while True:
    print("SERVER ACTIVE")
    sleep(5)