import sys
sys.path.append('/home/hwngnt/Code/pg')
from helper.draw import draw_connection, draw_keypoints
import cv2 
from threading import Thread 
from .Q import Q1, Q2, lm_list, vote, width, height, Stop


class Visualize :
    def __init__(self):
        self.action = ["Do Nothing", "Stop", "Move Straight", "Left turn", "Left turn waiting", "Right turn", "Lane Changing", "Slow down", "Pull over"]
        self.t = Thread(target=self.visualize, args=())
        self.t.daemon = True

    def start(self):
        self.t.start()

    def join(self):
        self.t.join()

    def visualize(self):
        print("[3]", Q2.qsize())
        while Q2.empty() is False:
        # if Q2.empty() is False:
            print("========================")
            self.vote              = vote
            self.lm                = lm_list([-1][:])
            self.frame             = Q2.get()         
            self.width             = width
            self.height            = height
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            position               = (10,100)
            fontScale              = (self.width * self.height) / (700 * 700)
            fontColor              = (255,0,0)
            thickness              = 2
            lineType               = 3

            self.frame = draw_keypoints(self.frame, self.lm, self.width, self.height)
            self.frame = draw_connection(self.frame, self.lm, self.width, self.height)
            print('[3] Q2',Q2.qsize())
            if self.vote == None:
                self.vote = 'loading...'
                cv2.putText(self.frame, str(self.vote), position, font,
                fontScale, fontColor, thickness, lineType)
            else:
                cv2.putText(self.frame,  str(self.action[self.vote.index(max(self.vote))]) + ": " + str(max(self.vote)), position, font,
                fontScale, fontColor, thickness, lineType)
            if Q2.qsize() == 0 and Stop!=False:
                print("[3]")
                continue
            elif Q2.qsize() == 0 and Stop!=True:
                print("[3] ========================")
                break
        # else:
        #     print('[3]',Q1.qsize())
            # print("!!!!!!!!!!!!!!")