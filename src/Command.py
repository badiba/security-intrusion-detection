from multiprocessing.connection import Client
import time

address = ('localhost', 6000)
conn = Client(address, authkey=b'secret password')
while (True):
    test = input()
    conn.send(test)

    msg = conn.recv()
    print(msg)

