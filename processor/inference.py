import collections
import torch
import numpy as np
import cv2
from st_gcn_aaai18 import ST_GCN_18
import sys
sys.path.append('/home/hwngnt/Code/pg')
from helper.draw import draw_connection, draw_keypoints
from helper.load_model import load_state, run_demo

import time


from helper.read import WebcamStream
from helper.process import ProcessQueue
from helper.visualize import Visualize
import mediapipe as mp
# model = torch.load('../checkpoints/pg_distance.pth')



# window = 60
# time_steps = 30
# stride = 5
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# video = cv2.VideoCapture(0)
# length = int(video.get(cv2. CAP_PROP_FRAME_COUNT))
# fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
# width= int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height= int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(width, height)
# lm_list = []
# x_list = []
# y_list = []
# keypoints = np.zeros((33), dtype=int)
# keypoints[0] = 1
# keypoints[11] = 1
# keypoints[12] = 1
# keypoints[13] = 1
# keypoints[14] = 1
# keypoints[15] = 1
# keypoints[16] = 1
# keypoints[23] = 1
# keypoints[24] = 1
# keypoints[25] = 1
# keypoints[26] = 1
# keypoints[27] = 1
# keypoints[28] = 1


# re_arrange_pose = [2,4,6,1,3,5,8,10,12,7,9,11,0]
video = WebcamStream(stream_id = 0)
video.get_size()
# process = ProcessQueue()
# visualize = Visualize()
video.start()
# process.start()
# visualize.start()
video.join()
# process.join()
# visualize.join()

# while True :
#     # if video.stopped is True :
#     #     break
#     # else :
#     Q = video.read()
#     # print('first', Q) 
#     # # new_frame_time = time.time()
#     process.processing()
#     visualize.visualize()
    # vote, lm, frame = process.processing(frame, process_queue)
    # visualize.visualize(vote, lm, frame, width, height)
    # fps = 1/(new_frame_time-prev_frame_time)
    # prev_frame_time = new_frame_time
    # fps = int(fps)
    # print(fps)
    # cv2.imshow('frame' , frame)
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     break
# video.stop() # stop the webcam stream 





# sleep(5)

# closing all windows 
# cv2.destroyAllWindows()




# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while video.isOpened():
#         flag, frame = video.read()
#         if not flag:
#             break
        
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
      
#         # Make detection
#         results = pose.process(image)
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         # Extract landmarks
#         landmarks = results.pose_landmarks.landmark
#         for id, lm in enumerate(landmarks):
#             if keypoints[id] == 1:
#                 x_list.append(lm.x)
#                 y_list.append(lm.y)
        
#         temp.append(x_list)
#         temp.append(y_list)
#         for i in re_arrange_pose:
#             # print(i)
#             new_pose[0].append(round(temp[0][i], 4))
#             new_pose[1].append(round(temp[1][i], 4))
#         lm_list.append(new_pose)
#         new_pose = [[],[]]
#         temp = []
        

#         x_list = []
#         y_list = []
#         queue.append(image)
#         if len(queue) == window:
#             lm_list = np.array(lm_list)
#             vote = [0,0,0,0,0,0,0,0,0]
#             for i in range(0, len(queue)-time_steps+1, stride):
#                 frames_X = []
#                 frames_Y = []
#                 for j in range(i, i+time_steps):
#                     frames_X.append(lm_list[j][0][:])
#                     frames_Y.append(lm_list[j][1][:])
#                 sample = [frames_X, frames_Y]
#                 sequence = np.array(sample)

#                 sequence = np.expand_dims(sequence, axis = 0)
#                 sequence = np.expand_dims(sequence, axis = -1)
#                 sequence = torch.Tensor(sequence).to(device)
#                 net = ST_GCN_18(in_channels=2, num_class=9, 
#                 graph_cfg={'layout': 'police',
#                         'strategy': 'distance'}).to(device)
#                 load_state(net, model)
#                 # print('Sequence: ',sequence.shape, sequence[0, :, -1, :, :])
#                 result = run_demo(net, sequence)
#                 prob = result.detach()
#                 vote[int(torch.argmax(prob))] += (torch.max(prob)).cpu().numpy()
#             queue[-1] = draw_keypoints(queue[-1], lm_list[-1][:], width, height)
#             queue[-1] = draw_connection(queue[-1], lm_list[-1][:], width, height)
#             cv2.putText(queue[-1], str(vote.index(max(vote))), 
#             bottomLeftCornerOfText, 
#             font, 
#             fontScale,
#             fontColor,
#             thickness,
#             lineType)

#             cv2.imshow('frame', queue[-1])
#             lm_list = list(lm_list)
#             lm_list.pop(0)
#             queue.pop(0)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     video.release()
#     # Destroy all the windows
#     cv2.destroyAllWindows()


