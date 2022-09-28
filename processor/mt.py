import multiprocessing
import cv2
import time
from queue import Queue
import numpy as np
import mediapipe as mp
import sys
sys.path.append('/home/hwngnt/Code/pg')
from helper.draw import draw_connection, draw_keypoints
from helper.load_model import load_state, run_demo
from processor.st_gcn_aaai18 import ST_GCN_18
import torch


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_frame(q1, q2, width, height):
    count = 0
    video = cv2.VideoCapture(0)
    width      = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    while True :
        start = time.time()
        grabbed, frame = video.read()
        q1.put(frame)
        count += 1
        # print("[1] Push success", q1.qsize())
        end = time.time()
        print('[1] time:', end - start)
        if count == 90:
            break
        
    video.release()
    cv2.destroyAllWindows()

def processing(q1, q2, lm_list):
    keypoints = np.zeros((33), dtype=int)
    keypoints[0] = 1
    keypoints[11] = 1
    keypoints[12] = 1
    keypoints[13] = 1
    keypoints[14] = 1
    keypoints[15] = 1
    keypoints[16] = 1
    keypoints[23] = 1
    keypoints[24] = 1
    keypoints[25] = 1
    keypoints[26] = 1
    keypoints[27] = 1
    keypoints[28] = 1
    window = 60
    time_steps = 30
    stride = 5
    re_arrange_pose = [2,4,6,1,3,5,8,10,12,7,9,11,0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('../checkpoints/pg_distance.pth')
    while True:
        if q1.empty() is False:
            start = time.time()
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                frame = q1.get()
                x_list = []
                y_list = []
                temp = []
                new_pose = [[],[]]
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                for id, lm in enumerate(landmarks):
                    if keypoints[id] == 1:
                        x_list.append(lm.x)
                        y_list.append(lm.y)

                temp.append(x_list)
                temp.append(y_list)
                for i in re_arrange_pose:
                    new_pose[0].append(round(temp[0][i], 4))
                    new_pose[1].append(round(temp[1][i], 4))

                lm_list.append(new_pose)
                new_pose = [[],[]]
                temp = []
                x_list = []
                y_list = []
                
                if len(lm_list) == window:
                    landmark = lm_list
                    landmark = np.array(landmark)
                    vote = [0,0,0,0,0,0,0,0,0]
                    
                    for i in range(0, len(landmark) - time_steps + 1, stride):
                        frames_X = []
                        frames_Y = []
                        for j in range(i, i+time_steps):
                            frames_X.append(landmark[j][0][:])
                            frames_Y.append(landmark[j][1][:])
                        sample = [frames_X, frames_Y]
                        sequence = np.array(sample)
                        
                        sequence = np.expand_dims(sequence, axis = 0)
                        sequence = np.expand_dims(sequence, axis = -1)
                        sequence = torch.Tensor(sequence).to(device)
                        net = ST_GCN_18(in_channels=2, num_class=9, 
                        graph_cfg={'layout': 'police',
                                'strategy': 'distance'}).to(device)
                        load_state(net, model)
                        result = run_demo(net, sequence)
                        prob = result.detach()
                        vote[int(torch.argmax(prob))] += (torch.max(prob)).cpu().numpy()
                    lm_list.pop(0)
                    q2.put(frame)
                    print('[2] q2',q2.qsize())
                else:
                    vote = None
                if q1.empty() is True:
                    break
            end = time.time()

            # print('[2] q1 size:', q1.qsize())
            print('[2] time:', end -start)
def visualize(q1, q2, lm_list, width, height):
    action = ["Do Nothing", "Stop", "Move Straight", "Left turn", "Left turn waiting", "Right turn", "Lane Changing", "Slow down", "Pull over"]
    
    while True:
        if q2.empty() is False:
            start = time.time()
            lm                = lm_list[-1][:]
            frame             = q2.get()
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            position               = (10,100)
            fontScale              = (width * height) / (700 * 700)
            fontColor              = (255,0,0)
            thickness              = 2
            lineType               = 3
            
            frame = draw_keypoints(frame, lm, width, height)
            frame = draw_connection(frame, lm, width, height)

            print('[3] q2:',q2.qsize())
            cv2.imshow('frame' , frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                q1 = Queue()
                break
            if q2.empty() is True and q1.empty() is True:
                break
            end = time.time()
            print('[3] time:', end-start)
        # else:
        #     continue

if __name__ == "__main__":
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    manager = multiprocessing.Manager()
    lm_list = manager.list()
    ns = manager.Namespace()
    ns.width = 0
    ns.height = 0    


    t1 = multiprocessing.Process(target=get_frame, args=(q1, q2, ns.width, ns.height))
    t2 = multiprocessing.Process(target=processing, args=(q1, q2, lm_list))
    t3 = multiprocessing.Process(target=visualize, args=(q1, q2, lm_list, ns.width, ns.height))

    t1.start()
    t2.start()
    t3.start()

    q1.close()
    q2.close()

    q1.join_thread()
    q2.join_thread()
    
    t1.join()
    t2.join()
    t3.join()