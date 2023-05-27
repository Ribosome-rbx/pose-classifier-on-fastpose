import pandas as pd
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch


datapath = os.path.join(os.getcwd(), 'pose_dataset')
label_names = ['creep_forward', 'creep_left', 'creep_right', \
               'dance', 'jump', 'punch', 'run_forward', \
               'run_left', 'run_right', 'stand_still',  \
               'walk_forward', 'walk_left', 'walk_right']

# config
train_data_ratio = 0.8
batch_size = 32

# 需要/std吗, data unbalance
def normalized_frame(frame):
    frame = frame.reshape(-1, 2)
    frame = (frame - frame.mean(axis=0)) / frame.std(axis=0)
    return frame.flatten()

def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)



traindata = []
testdata = []
labels = []
#  load train data
for i, filename in enumerate(label_names):
    if label_names == 'test' : continue
    file = load(datapath + '/' + filename + ".pickle")
    for j in range(len(file)):
        file[j] = normalized_frame(file[j])
    for j in range(len(file) - 10):
        ten_gram = file[j:j+10]
        ten_gram = np.array(ten_gram)
        traindata.append(ten_gram.flatten())
        labels.append(i)

train_data = torch.tensor(np.array(traindata))
train_labels = torch.tensor(np.array(labels), dtype=torch.long)

#  load test data
filename = "test.pickle"
file = load(datapath + '/' + filename)
for j in range(len(file)):
    file[j] = normalized_frame(file[j])
for j in range(len(file) - 10):
    ten_gram = file[j:j+10]
    ten_gram = np.array(ten_gram)
    testdata.append(ten_gram.flatten())
test_data = torch.tensor(np.array(testdata))

class KNN:
    def __init__(self, k, train_data, train_labels):
        self.k = k
        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, test_data):
        distances = torch.cdist(test_data, self.train_data)
        _, indices = torch.topk(distances, k=self.k, largest=False)
        knn_labels = self.train_labels[indices]
        predicted_labels, _ = torch.mode(knn_labels, dim=1)
        return predicted_labels
    
# Create a KNN instance with k=3
knn = KNN(k=3, train_data=train_data, train_labels=train_labels)

if __name__ == "__main__":

    # Make predictions
    predictions = knn.predict(test_data)

    # Print the predicted labels
    for i in predictions[:10]:
        print(label_names[i],"\n")

