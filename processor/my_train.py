import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from Dataset import Dataset
from st_gcn_aaai18 import ST_GCN_18
from torchmetrics.functional import accuracy
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import time

start = time.time()
def set_seed(seed=2022):
    # Set seed for each run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_seed()

params = {
    'batch_size': 32,
    'shuffle'   : True,
    'num_workers': 6
}

train = Dataset('../data/police/data_v2_train.npy', '../data/police/data_v2_train_label.npy')
training_generator = DataLoader(train, **params)

val = Dataset('../data/police/data_v2_val.npy', '../data/police/data_v2_val_label.npy')
validation_generator = DataLoader(val, **params)

test = Dataset('../data/police/data_v2_test.npy', '../data/police/data_v2_test_label.npy')
testing_generator = DataLoader(test, **params)

num_epochs = 20
learning_rate = 5e-4



model = ST_GCN_18(in_channels=2, num_class=9, 
graph_cfg={'layout': 'police',
        'strategy': 'distance'}).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)

# classes = ['Stand in attention', 'stop', 'move straight', 'left turn', 'left turn waiting', 'right down', 'lane changing', 'slow down', 'pullover']
# loss_train = []
# loss_val = []
loss_per_epoch_train = []
loss_per_epoch_val = []
train_accuracy_per_epoch = []
val_accuracy_per_epoch = []
# epochs = []
# train_accuracy = []
# val_accuracy = []
# best_acc = 0
# best_loss = 0

# y_pred = []
# y_true = []


for epoch in range(num_epochs):
    with torch.set_grad_enabled(True):
        model.train()
        for batch, labels in training_generator:
            train_batch, train_labels = batch.to(device), labels.to(device)
            outputs = model(train_batch)
            optimizer.zero_grad()
            loss = criterion(outputs, train_labels)
            loss_per_epoch_train.append(loss.item())
            pred_outputs = torch.softmax(outputs, dim=1).to(device)
            pred_outputs_index = pred_outputs.argmax(1).to(device)
            prediction = torch.zeros(pred_outputs.shape, dtype=torch.int).to(device).scatter(1, pred_outputs_index.unsqueeze(1).to(device), 1.0)
            train_acc = accuracy(prediction, train_labels.to(torch.int), subset_accuracy=True).item()
            train_accuracy_per_epoch.append(train_acc)
            loss.backward()
            optimizer.step()
    with torch.set_grad_enabled(False):
        model.eval()
        for batch, labels in validation_generator:
            val_batch, val_labels = batch.to(device), labels.to(device)
            val_outputs = model(val_batch)
            val_loss = criterion(val_outputs, val_labels)
            pred_outputs = torch.softmax(val_outputs, dim=1).to(device)
            pred_outputs_index = pred_outputs.argmax(1).to(device)
            prediction = torch.zeros(pred_outputs.shape, dtype=torch.int).to(device).scatter(1, pred_outputs_index.unsqueeze(1).to(device), 1.0)
            val_acc = accuracy(prediction, val_labels.to(torch.int), subset_accuracy=True).item()
            val_accuracy_per_epoch.append(val_acc)
            loss_per_epoch_val.append(val_loss.item())

    # break
    scheduler.step()
    if epoch % 1 == 0:
        # epochs.append(epoch)
        # loss_train.append(sum(loss_per_epoch_train)/len(loss_per_epoch_train))
        # loss_val.append(sum(loss_per_epoch_val)/len(loss_per_epoch_val))
        print("Train Epoch: %d, loss: %1.5f" % (epoch, sum(loss_per_epoch_train)/len(loss_per_epoch_train)))
        print("Val Epoch: %d, val_loss: %1.5f" % (epoch, sum(loss_per_epoch_val)/len(loss_per_epoch_val)))

        print('Train Accuracy:', sum(train_accuracy_per_epoch)/len(train_accuracy_per_epoch))
        print('Val Accuracy:', sum(val_accuracy_per_epoch)/len(val_accuracy_per_epoch))
        # train_accuracy.append(sum(train_accuracy_per_epoch)/len(train_accuracy_per_epoch))
        # val_accuracy.append(sum(val_accuracy_per_epoch)/len(val_accuracy_per_epoch))
        # save best model
        # if sum(val_accuracy_per_epoch)/len(val_accuracy_per_epoch) > best_acc:
        #     best_acc = sum(val_accuracy_per_epoch)/len(val_accuracy_per_epoch)
        #     print('best model at {}'.format(best_acc))
        #     torch.save(model.state_dict(), '../checkpoints/pg_distance.pth')
        train_accuracy_per_epoch = []
        val_accuracy_per_epoch = []
        loss_per_epoch_train = []
        loss_per_epoch_val = []

end = time.time()
print("=================TIME===============", end - start)
# print(best_acc)
# print(best_loss)
# plt.plot(epochs, loss_train, 'g', label='Training loss')
# plt.plot(epochs, loss_val, 'b', label='validation loss')
# plt.title('loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('../media/police_loss_distance_data_v2_uniform.jpg')
# plt.clf()
# plt.plot(epochs, train_accuracy, 'g', label='Training accuracy')
# plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
# plt.title('Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('../media/police_acc_distance_data_v2_uniform.jpg')
# plt.clf()



# print(y_true_train)
# print(y_true_val)
# y_true_train = y_true_train.cpu()

# cf_matrix = confusion_matrix(y_true_train, y_pred_train)
# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *9, index = [i for i in classes],
#                      columns = [i for i in classes])
# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.savefig('../media/cm_train.png')
# plt.clf()

# y_true_val = torch.Tensor(y_true_val)
# y_pred_val = torch.Tensor(y_pred_val)

# for i in range (len(y_true_val)):
#     y_true_val[i] = y_true_val[i].cpu()
#     y_true_val[i] = torch.Tensor(y_true_val[i])

# for i in range (len(y_pred_val)):
#     y_pred_val[i] = y_pred_val[i].cpu()
#     y_pred_val[i] = torch.Tensor(y_pred_val[i])




# print(type(y_pred_val[0]))
# print(type(y_true_val[0]))
# print(len(y_pred_val))
# print(len(y_true_val))


# new_true = []
# for i in y_true_val:
#     for j in i:
#         new_true.append(j)

# new_pred = []
# for i in y_pred_val:
#     for j in i:
#         new_pred.append(j)

# print(len(new_pred))
# print(len(new_true))

# for i in y_pred_val:
#     for j in i:


# y_true_val = y_true_val.cpu()
# y_pred_val = y_pred_val.cpu()
# cf_matrix = confusion_matrix(new_true[:20000], new_pred[:20000])
# print(cf_matrix)
# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *9, index = [i for i in classes],
#                      columns = [i for i in classes])
# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.savefig('../media/cm_val.png')

