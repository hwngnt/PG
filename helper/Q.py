from multiprocessing.resource_sharer import stop
from queue import Queue
global Q1
Q1 = Queue()

global Q2
Q2 = Queue()
    
global lm_list
lm_list = []

global vote 
vote = [0,0,0,0,0,0,0,0,0]

global width, height


global Stop
Stop = False

width = 0
height = 0