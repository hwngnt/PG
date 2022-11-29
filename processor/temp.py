import enum
import numpy as np
import cv2
from Dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from st_gcn_aaai18 import ST_GCN_18
import collections

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def load_state(net, checkpoint):
#     source_state = checkpoint
#     target_state = net.state_dict()
#     new_target_state = collections.OrderedDict()
#     for target_key, target_value in target_state.items():
#         if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
#             new_target_state[target_key] = source_state[target_key]
#         else:
#             new_target_state[target_key] = target_state[target_key]
#             print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

#     net.load_state_dict(new_target_state)
    
# def run_demo(net, data):
#     net = net.eval()
#     output = net(data)
#     output =  torch.softmax(output, dim=1)
#     return output

# params = {
#     'batch_size': 32,
#     'shuffle'   : True,
#     'num_workers': 6
# }
# test = Dataset('../data/police/data_v2_test.npy', '../data/police/data_v2_test_label.npy')
# testing_generator = DataLoader(test, **params)


# model = torch.load('../checkpoints/pg_uniform.pth')
# net = ST_GCN_18(in_channels=2, num_class=9, 
#             graph_cfg={'layout': 'police',
#                     'strategy': 'uniform'}).to(device)
# load_state(net, model)

# y_true = []
# y_pred = []


# net = net.eval()

# classes = ['Stand in attention', 'stop', 'move straight', 'left turn', 'left turn waiting', 'right down', 'lane changing', 'slow down', 'pullover']
# for batch, labels in testing_generator:
#     test_batch, test_labels = batch.to(device), labels.to(device)
#     test_outputs = net(test_batch)
#     pred_outputs = torch.softmax(test_outputs, dim=1).to(device)
#     pred_outputs_index = pred_outputs.argmax(1).to(device)
#     prediction = torch.zeros(pred_outputs.shape, dtype=torch.int).to(device).scatter(1, pred_outputs_index.unsqueeze(1).to(device), 1.0)
    
#     y_pred.append(pred_outputs_index)
#     y_true.append(test_labels.argmax(1))
#     # break

# for i in range(len(y_pred)):
#     y_pred[i] = y_pred[i].to("cpu")
# for i in range(len(y_true)):
#     y_true[i] = y_true[i].to("cpu")
# # print(y_pred[0])
# # print(y_true[0])

# new_true = []
# new_pred = []

# for i in y_pred:
#     new_pred.extend(i.tolist())

# for i in y_true:
#     new_true.extend(i.tolist())
# print(len(new_pred))
# print(len(new_true))
# cf_matrix = confusion_matrix(new_true, new_pred)

# count=0
# for i in new_true:
#     if i == 0:
#         count+=1

# print("true", count)
# count = 0 
# for i in new_pred:
#     if i==0:
#         count+=1
# print("pred", count)
# print(cf_matrix)
# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *9, index = [i for i in classes],
#                      columns = [i for i in classes])
# df_cm = pd.DataFrame(cf_matrix/cf_matrix.sum(1), index = [i for i in classes],
#                      columns = [i for i in classes])
# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.savefig('../media/cm_v1_uni.png')

# y_pred = torch.Tensor(y_pred, device='cpu')
# y_true = torch.Tensor(y_true, device='cpu')

# y_pred = np.array(y_pred)
# y_true = np.array(y_true)

# print(y_pred.shape)
# print(y_true.shape)





# classes = ['stand in action', 'stop', 'move straight', 'left turn', 'left turn waiting', 'right down', 'lane changing', 'slow down', 'pullover']
# train_label = np.load("/home/asilla/hungnt/my_GCN/data/police/data_v1_label.npy")
# val_label = np.load("/home/asilla/hungnt/my_GCN/data/police/data_v1__val_label.npy")
# train = np.load("/home/asilla/hungnt/my_GCN/data/police/data_v1.npy")
# val = np.load("/home/asilla/hungnt/my_GCN/data/police/data_v1_val.npy")


# a = np.concatenate((train, val))
# print(a.shape)
# b = np.concatenate((train_label, val_label))
# print(b.shape)
# new_train = []
# new_val = []
# new_test = []
# new_train_lb = []
# new_val_lb = []
# new_test_lb = []

# for i in range(len(a)):
#     if len(new_train) < len(a)/10*6:
#         new_train.append(a[i])
#         new_train_lb.append(b[i])
#     elif len(new_val) < len(a)/10*2:
#         new_val.append(a[i])
#         new_val_lb.append(b[i])
#     else:
#         new_test.append(a[i])
#         new_test_lb.append(b[i])

# new_train, new_val, new_train_lb, new_val_lb = train_test_split(a, b, test_size=0.4, random_state=42)

# new_val, new_test, new_val_lb, new_test_lb = train_test_split(new_val, new_val_lb, test_size=0.5, random_state=42)

# new_train = np.array(new_train)
# new_val = np.array(new_val)
# new_test = np.array(new_test)

# new_train_lb = np.array(new_train_lb)
# new_val_lb = np.array(new_val_lb)
# new_test_lb = np.array(new_test_lb)


# print(new_train.shape)
# print(new_val.shape)
# print(new_test.shape)

# print(new_train_lb.shape)
# print(new_val_lb.shape)
# print(new_test_lb.shape)

# np.save('../data/police/data_v2_train.npy', new_train)
# np.save('../data/police/data_v2_test.npy', new_test)
# np.save('../data/police/data_v2_val.npy', new_val)
# np.save('../data/police/data_v2_train_label.npy', new_train_lb)
# np.save('../data/police/data_v2_test_label.npy', new_test_lb)
# np.save('../data/police/data_v2_val_label.npy', new_val_lb)

    

# data = np.load("../data/police/data_v2_train.npy")
label = np.load("../data/police/data_v2_val_label.npy")
# new_data = []
# new_label =[]
# print(data.shape)
# print(label.shape)
# count0 = 0
# count1 = 0
# count2 = 0
# count3 = 0
# count4 = 0
# count5 = 0
# count6 = 0
# count7 = 0
# count8 = 0
# for index,i in enumerate(label):
#     if str(i) == "[1 0 0 0 0 0 0 0 0]":
#         # print(data[index].shape)
#         count0 += 1
#     elif str(i) == "[0 1 0 0 0 0 0 0 0]":
#         count1 += 1
#     elif str(i) == "[0 0 1 0 0 0 0 0 0]":

#         count2 += 1
#     elif str(i) == "[0 0 0 1 0 0 0 0 0]":

#         count3 += 1
#     elif str(i) == "[0 0 0 0 1 0 0 0 0]":
#         count4 += 1
#     elif str(i) == "[0 0 0 0 0 1 0 0 0]":

#         count5 += 1
#     elif str(i) == "[0 0 0 0 0 0 1 0 0]":

#         count6 += 1
#     elif str(i) == "[0 0 0 0 0 0 0 1 0]":

#         count7 += 1
#     elif str(i) == "[0 0 0 0 0 0 0 0 1]":

#         count8 += 1

# amount = [count0, count1, count2, count3, count4, count5, count6, count7, count8]
# print(amount)
# label = np.array(classes)
# # amount  = np.array(amount)
# # print(amount)
# plt.figure(figsize=(10, 3))  # width:20, height:3
# plt.barh(label, amount)
# plt.savefig('../media/data_val_v2_distribution.png')
# # print(count0)
# new_data = np.array(new_data)
# print(new_data.shape)
# new_label = np.array(new_label)
# print(new_label.shape)
# # print(new_data[0][0])
# np.save('../data/police/data_v1_val.npy', new_data)
# np.save('../data/police/data_v1__vallabel.npy', new_label)

# print(count1)
# print(count2)
# print(count3)
# print(count4)
# print(count5)
# print(count6)
# print(count7)
# print(count8)


# import sys
# sys.path.append('/home/asilla/hungnt/my_GCN')
# from helper.draw import draw_connection, draw_keypoints
# video = cv2.VideoCapture('../media/vid.mp4')
# length = int(video. get(cv2. CAP_PROP_FRAME_COUNT))
# fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
# width= int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height= int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# videoWriter = cv2.VideoWriter('../media/vid_1.avi', fourcc, 15.0, (width,height))
# frame_count = 0
# while video.isOpened():
#     flag, frame = video.read()
#     if not flag:
#         break

#     videoWriter.write(frame)
#     frame_count += 1
#     if frame_count == 30:
#         break
# video.release()
# videoWriter.release()