import cv2 
import time 
from threading import Thread 
from queue import Queue
import numpy as np
import mediapipe as mp
import sys
sys.path.append('/home/hwngnt/Code/pg')
from helper.draw import draw_connection, draw_keypoints
from helper.load_model import load_state, run_demo
from processor.st_gcn_aaai18 import ST_GCN_18
import torch
from .Q import Q1, Q2, lm_list, vote, Stop

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



class ProcessQueue:
    def __init__(self, window=60, time_steps = 30, stride=5):
        self.window = window 
        self.time_steps = time_steps
        self.stride = stride
        self.img_list = []
        self.keypoints = np.zeros((33), dtype=int)
        self.keypoints[0] = 1
        self.keypoints[11] = 1
        self.keypoints[12] = 1
        self.keypoints[13] = 1
        self.keypoints[14] = 1
        self.keypoints[15] = 1
        self.keypoints[16] = 1
        self.keypoints[23] = 1
        self.keypoints[24] = 1
        self.keypoints[25] = 1
        self.keypoints[26] = 1
        self.keypoints[27] = 1
        self.keypoints[28] = 1


        self.re_arrange_pose = [2,4,6,1,3,5,8,10,12,7,9,11,0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load('../checkpoints/pg_distance.pth')
        self.t = Thread(target=self.processing, args=())
        self.t.daemon = True

    def start(self):
        self.t.start()

    def join(self):
        self.t.join()

    def processing(self):
        print('=======',Q1.qsize())
        while Q1.empty() is False:
        # if Q1.empty() is False:
            # print("[2]", Q1.qsize())
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                self.frame = Q1.get()
                print('[2] Q1',Q1.qsize())  
                x_list = []
                y_list = []
                temp = []
                new_pose = [[],[]]
                # Recolor image to RGB
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                for id, lm in enumerate(landmarks):
                    if self.keypoints[id] == 1:
                        x_list.append(lm.x)
                        y_list.append(lm.y)

                temp.append(x_list)
                temp.append(y_list)
                for i in self.re_arrange_pose:
                    new_pose[0].append(round(temp[0][i], 4))
                    new_pose[1].append(round(temp[1][i], 4))

                # print(lm_list)
                lm_list.append(new_pose)
                # # print(self.lm_list)
                new_pose = [[],[]]
                temp = []
                x_list = []
                y_list = []
                
                if len(lm_list) == self.window:
                    # print(len(lm_list))
                    landmark = lm_list
                    landmark = np.array(landmark)
                    vote = [0,0,0,0,0,0,0,0,0]
                    # print(landmark.shape)
                    for i in range(0, len(landmark) - self.time_steps + 1, self.stride):
                        frames_X = []
                        frames_Y = []
                        for j in range(i, i+self.time_steps):
                            frames_X.append(landmark[j][0][:])
                            frames_Y.append(landmark[j][1][:])
                        sample = [frames_X, frames_Y]
                        sequence = np.array(sample)

                        sequence = np.expand_dims(sequence, axis = 0)
                        sequence = np.expand_dims(sequence, axis = -1)
                        sequence = torch.Tensor(sequence).to(self.device)
                        net = ST_GCN_18(in_channels=2, num_class=9, 
                        graph_cfg={'layout': 'police',
                                'strategy': 'distance'}).to(self.device)
                        load_state(net, self.model)
                        result = run_demo(net, sequence)
                        prob = result.detach()
                        vote[int(torch.argmax(prob))] += (torch.max(prob)).cpu().numpy()
                    lm_list.pop(0)
                    Q2.put(self.frame)
                    print('[2] Q2',Q2.qsize())
                else:
                    vote = None
            if Q1.empty() and Stop!=True:
                # print("[2] =======")
                continue
            elif Q1.empty() and Stop == True:
                break
        # else:
        #     print("Q1 is empty")