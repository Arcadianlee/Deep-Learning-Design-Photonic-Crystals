#Code for the validation of trained model on Q
#Apr 2021 Renjie Li, NOEL @ CUHK SZ

import torch 
import torchvision
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim 
import pandas as pd
import numpy as np 
import h5py
import torchvision.transforms as transforms
from datetime import datetime

class TensorsDataset(torch.utils.data.Dataset):
    
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''

    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

#read data from mat file
print("loading the mat")
f = h5py.File('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/L3_dataset/Input_v.mat','r')
data = f['Input']
Input = np.array(data) # For converting to a NumPy array

f = h5py.File('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/L3_dataset/Output_v.mat','r')
data = f['QnV']
Output = np.array(data) # For converting to a NumPy array

print("converting to tensor")
input_tensor = torch.tensor(Input)
output_tensor = torch.tensor(Output) 

#swap the axes
input_tensor = input_tensor.permute(3,2,1,0).float()
output_tensor = output_tensor.permute(1,0).float()
output_tensor = output_tensor[:,0] #do Q first
output_tensor = output_tensor.view(-1,1) #correct the dimension

print(output_tensor[-1])

print(input_tensor.shape)
print(output_tensor.shape)

#produce the full dataset
transformer=transforms.Normalize(mean=[-8.7270e-13,3.3969e-13,-1.6978e-12], std=[0.0000000005,0.0000000005,0.0000000005])

dataset=TensorsDataset(input_tensor, output_tensor,transforms=transformer)

#split into training and test datasets
train_size = 0
test_size = len(output_tensor)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#load the data
test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(output_tensor), shuffle=False)

#set up the network
#create a class for the CNN
class Net(nn.Module):
    #build the network (cnn+fc)
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,20, kernel_size=(3,3), padding = 1, bias=False)
        self.bn1=nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20,40,kernel_size=(3,3),bias=False)
        self.bn2=nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(240,120)
        self.fc2 = nn.Linear(120,50)
        self.fc3 = nn.Linear(50,1)
    #pass data to the CNN. x represents the data    
    def forward(self,x):
        x = F.relu(F.avg_pool2d(self.conv1(x),(1,2)))
#         print(x.shape)
        x = F.relu(F.avg_pool2d(self.conv2(x),(1,2)))
#        print(x.shape)

        x = x.view(x.size(0),-1)
#        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = Net()
network_state_dict = torch.load('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/L3_model.pt')
network.load_state_dict(network_state_dict)


test_losses = [] #for Q
testV_losses = [] #for V
test_output = []  
testV_output = []
test_target = []
testV_target = []
pred_error = []

#test loop
def test():
    #global test_output
    network.eval()
    with torch.no_grad(): #disable the gradient computation
        for data, target in test_loader:
            output = network(data)
            #save the test result
            #Q
            test_output.append(output)
            test_target.append(target)

            pred_err = 100*torch.abs((output - target))/target
            pred_error.append(pred_err)

            #print('pred errors...')
            #print(pred_err)


start=datetime.now()

for epoch in range(0,1):

    test()
    #print('Q predicted/true values...')
    #print(test_output,test_target)

print((datetime.now()-start))

#convert from list to tensor
pred_errorT = torch.cat(pred_error,0)
pred_errorA = pred_errorT.numpy()

print(min(pred_errorA))

red_square = dict(markerfacecolor='r', marker='s')
fig, ax = plt.subplots()
ax.boxplot(pred_errorA, flierprops=red_square, vert=False)
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/valid_boxplot.eps')

fig = plt.figure()
plt.hist(pred_errorA, 50, density=False)
plt.xlim(-0.05,0.80)
plt.xticks(np.arange(0, 0.85, 0.05))
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/valid_hist.eps')

print(np.amin(pred_errorA),np.mean(pred_errorA), np.median(pred_errorA))