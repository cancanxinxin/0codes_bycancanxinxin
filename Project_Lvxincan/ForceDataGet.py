#!/usr/bin/python3
# sudo chmod 666 /dev/ttyACM0
import serial
import time
import threading
import optoforce
import binascii

SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUDRATE = 1000000
sensor_type = "s-ch/6-axis"
starting_index = 0
scaling_factors = 0

class UART_Receiver(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.sp = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0.5)
        self._wrenches = []

    def readSerial(self):
        return self.sp.readline()

    def run(self):
        self.sp.reset_input_buffer()
        while True: 
            if (self.sp.in_waiting > 0): 
                data = self.readSerial()
                print('the data is:',data)
            time.sleep(0.5)

def getForceData():
    test = optoforce.OptoforceDriver('/dev/ttyACM0',"s-ch/6-axis",
        [[41.1398,41.0959,7.60483,1158,1082,1513.5]],starting_index = 0)
    test.config("100Hz","15Hz",zero = False)
    while True:
        # print(getForceData())
        data = test.read()
        if isinstance(data, optoforce.OptoforceData):
            # print(data) 
            for i in range(test.nb_sensors()):
                # print('fx=',float(data.force[i][0]))
                # print('fy=',float(data.force[i][1]))
                # print('fz=',float(data.force[i][2]))
                # print('Tx=',float(data.force[i][3]))
                # print('Ty=',float(data.force[i][4]))
                # print('Tz=',float(data.force[i][5]))
                return(data.force[i][0:6])

if __name__ == '__main__':
    # sp = UART_Receiver()
    # sp.run()
    # _publishers = []
    # _wrenches = []
    print(getForceData())

    '''
    following is OK
    '''
    # test = optoforce.OptoforceDriver('/dev/ttyACM0',"s-ch/6-axis",[[41.1398,41.0959,7.60483,1158,1082,1513.5]],starting_index = 0)
    # test.config("100Hz","15Hz",zero = False)
    # while True:
    #     # print(getForceData())
    #     data = test.read()
    #     if isinstance(data, optoforce.OptoforceData):
    #         # print(data)
    #         for i in range(test.nb_sensors()):
    #             print('fx=',float(data.force[i][0]))
    #             print('fy=',float(data.force[i][1]))
    #             print('fz=',float(data.force[i][2]))
    #             print('Tx=',float(data.force[i][3]))
    #             print('Ty=',float(data.force[i][4]))
    #             print('Tz=',float(data.force[i][5]))


