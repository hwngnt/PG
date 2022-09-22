import cv2 
import time 
from threading import Thread 
from queue import Queue
from .Q import Q1, width, height, Stop



class WebcamStream :
    def __init__(self, stream_id=0):
        self.stream_id = stream_id 
        
        self.vcap      = cv2.VideoCapture(self.stream_id)
        
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
            
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)
        
        self.stopped = True 

        self.t = Thread(target=self.update, args=())
        # self.Q = Queue()
        # self.Q.copy = Queue()
        self.t.daemon = True

    def get_size(self):
        width      = int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(width, height)

    def start(self):
        self.stopped = False
        self.t.start()
        
    def join(self):
        self.t.join()

    def update(self):
        count = 0
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            # self.Q.put(self.frame)
            # self.Q.copy.put(self.frame)
            # print(Q1.qsize())
            Q1.put(self.frame)
            count += 1
            print("[1] Push success", Q1.qsize())
            # print(self.frame.shape)
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
            # if count == 90:
            #     Stop = True
            #     break
        self.vcap.release()

    def read(self):
        return Q1.qsize()

    def stop(self):
        self.stopped = True 
