from email.mime import image
import multiprocessing
import cv2
import os
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


def get_frame(q1, stop):
    count = 0
    video = cv2.VideoCapture('train_video.mp4')
    while True :
        # print('[1] stop:', hex(id(stop)))
        start = time.time()
        grabbed, frame = video.read()
        if grabbed == False : 
            stop[0] = 1
            break
        # time.sleep(0.1)
        q1.put(frame)
        count += 1
        print("[1] Push success", q1.qsize())
        end = time.time()
        # print('[1] time:', end - start)
        if stop[0] == 1:
            break
        # if count == 90:
        #     stop = False
        #     break
    
        
    video.release()
    cv2.destroyAllWindows()
    print('[1] ======================================== CLOSED  ===================================')
    os._exit(0)

def processing(q1, q2, lm_list, vote_list, stop, is_visible):
    mp_pose = mp.solutions.pose
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
    net = ST_GCN_18(in_channels=2, num_class=9, 
                        graph_cfg={'layout': 'police',
                                'strategy': 'distance'}).to(device)
    load_state(net, model)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while True:
        if q1.empty() is False:
            start = time.time()
            # print('[2] q1:', q1.qsize())
            # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
            # print(type(results.pose_landmarks))
            if results.pose_landmarks is not None:
                # print("[2] ===============================")
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
                # print('[2] new pose', new_pose)
                lm_list.append(new_pose)
                # print("[2] lm_list", len(lm_list))
                new_pose = [[],[]]
                temp = []
                x_list = []
                y_list = []
                is_visible[0] = 0
            else:
                is_visible[0] = 1
                # lm_list.append([[0]*13,[0]*13])
            q2.put(frame)
            # print('[2] q2', q2.qsize())
            if len(lm_list) == window:
                print('[2] lm_list', len(lm_list))
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
                    # print('{}-{}'.format(i, i+time_steps))
                    sequence = np.expand_dims(sequence, axis = 0)
                    sequence = np.expand_dims(sequence, axis = -1)
                    sequence = torch.Tensor(sequence).to(device)
                    result = run_demo(net, sequence)
                    prob = result.detach()
                    vote[int(torch.argmax(prob))] += (torch.max(prob)).cpu().numpy()
                vote_list.append({vote.index(max(vote)): max(vote)/7})
                # print(vote_list)
                lm_list.pop(0)
                # q2.put(frame)
                # print('[2] q2',q2.qsize())
                
            # else:
            #     vote = None
            # if q1.empty() is True and stop[0] == 1:
            #     break
            if stop[0] == 1:
                break
            end = time.time()
    print('[2] ======================================== CLOSED  ===================================')
    os._exit(0)
            # print('[2] q1 size:', q1.qsize())
            # print('[2] time:', end -start)
def visualize(q1, q2, lm_list, vote_list, stop, is_visible):
    action = ["Do Nothing", "Stop", "Move Straight", "Left turn", "Left turn waiting", "Right turn", "Lane Changing", "Slow down", "Pull over"]
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        if q2.empty() is False:
            if is_visible[0] == 0:
            # print('[3] stop', hex(id(stop)))
                new_frame_time = time.time()
                start = time.time()
                # print('[3] lm_list', lm_list)
                lm                     = lm_list[-1][:]
                frame                  = q2.get()
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                position               = (10,100)
                fontScale              = (frame.shape[1] * frame.shape[0]) / (700 * 700)
                fontColor              = (255,0,0)
                thickness              = 2
                lineType               = 3
                image = draw_keypoints(frame, lm)
                image = draw_connection(image, lm)
                if not vote_list:
                    vote = 'Loading...'
                    cv2.putText(image,  vote, position, font,
                        fontScale, fontColor, thickness, lineType)
                else:
                    vote                   = vote_list[-1]
                    action_name = [key for key in vote.keys()]
                    prob = [prob for prob in vote.values()]
                    cv2.putText(image,  str(action[action_name[0]]) + ": " + str(prob), position, font,
                        fontScale, fontColor, thickness, lineType)
                    vote_list.pop(0)
                # print(vote)
                # print('[3] frame ==========================================', frame.shape)
                # print('[3] lm:',lm)
                # print('[3] stop:', stop)
                # print('[3] frame ==========================================', lm)
                
                # print('[3] q2:',q2.qsize())
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)
                print('=========================FPS=========================',fps)
                # print(str(action[int(vote.keys())]) + ": " + str(vote.values()))
            # cv2.putText(image,  str(action[action_name[0]]) + ": " + str(prob), position, font,
            #         fontScale, fontColor, thickness, lineType)
            else:
                image                  = q2.get()
            cv2.imshow('frame' , image)
            # print('[3] ================================')
            key = cv2.waitKey(1)
            if key == ord('q'):
                # q1.close()
                # q2.close()
                # q1 = multiprocessing.Queue()
                # q2 = multiprocessing.Queue()
                stop[0] = 1
                # print('[3] stop', hex(id(stop)))
                print('===================================BREAK========================================')
                break
            # if q2.empty() is True and q1.empty() is True and stop[0] == 1:
                # break
            end = time.time()
            # print('[3] time:', end-start)
        # else:
        #     continue
    print('[3] ======================================== CLOSED  ===================================')
    os._exit(0)
    # sys.exit()

if __name__ == "__main__":
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    lm_list = manager.list()
    vote_list = manager.list()
    # stop = bool(multiprocessing.Value('i', False).value)
    stop = multiprocessing.Array('i', [0])
    is_visible = multiprocessing.Array('i', [0])

    # print(hex(id(stop)))
    # ns.lm_list = []

    t1 = multiprocessing.Process(target=get_frame, args=(q1, stop))
    t2 = multiprocessing.Process(target=processing, args=(q1, q2, lm_list, vote_list, stop, is_visible))
    t3 = multiprocessing.Process(target=visualize, args=(q1, q2, lm_list, vote_list, stop, is_visible))
    
    t1.daemon = True
    t2.daemon = True
    t3.daemon = True

    t1.start()
    t2.start()
    t3.start()

    q1.close()
    q2.close()

    # print('main process ==================')
    # q1.join_thread()
    # q2.join_thread()
    
    t1.join()
    print('[1] main process ==================')
    t2.join()
    print('[2] main process ==================')
    t3.join()
    print('[3] main process ==================')
