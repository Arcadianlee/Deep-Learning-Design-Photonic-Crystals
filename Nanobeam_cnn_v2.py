#Code for the deep learning of Nanobeam structure
#This current iteration includes Q factor and modal volume V with CNN regression
#October 2021 Renjie Li, NOEL @ CUHK SZ, version 2. 

# %%
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


#initialize hypermeters
n_epochs = 500
batch_size_train = 64
batch_size_test = 100
learning_rate = 0.01
momentum = 0.9
L2reg = 0.001
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


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
f = h5py.File('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/L3_dataset/Input_NB.mat','r')
data = f['Input']
Input = np.array(data) # For converting to a NumPy array

f = h5py.File('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/L3_dataset/Output_NB.mat','r')
data = f['QnV']
Output = np.array(data) # For converting to a NumPy array

print("converting to tensor")
input_tensor = torch.tensor(Input)
output_tensor = torch.tensor(Output) 

#swap the axes
input_tensor = input_tensor.permute(3,2,1,0).float()
output_tensor = output_tensor.permute(1,0).float()
#output_tensor = output_tensor[:,0] #do Q first
output_tensor = output_tensor.view(-1,2) #correct the dimension

print(output_tensor[-1])

print(input_tensor.shape)
print(output_tensor.shape)

#produce the full dataset
print(torch.std_mean(output_tensor[:,0])) #calculate mean and stddev
print(torch.std_mean(output_tensor[:,1]))

output_tensor[:,0] = (output_tensor[:,0]-4.8502e+00)/0.0922  #normalize Q
output_tensor[:,1] = (output_tensor[:,1]-1.9287e-03)/0.0002  #normalize V 

#input_tensor = (input_tensor-0)/0.0000000005  #normalize input

print(output_tensor[-1])
print(input_tensor[-1])

transformer=transforms.Normalize(mean=[0,0,0], std=[0.0000000005,0.0000000005,0.0000000005])

dataset=TensorsDataset(input_tensor, output_tensor,transforms=transformer)

#transformer_target=transforms.Normalize(mean=[4.8502e+00, 1.9287e-03], std=[0.0922, 0.0002])

#split into training and test datasets
train_size = int(0.8 * len(dataset)) #80% of dataset for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


#load the data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape,example_targets.shape)


#set up the network
#create a class for the CNN
class Net(nn.Module): 
    #build the network (cnn+fc)
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,20, kernel_size=(2,3), padding = 1, bias=False)
        self.conv2 = nn.Conv2d(20,40,kernel_size=(1,3), bias=False)
        self.fc1 = nn.Linear(160,120)
        self.fc2 = nn.Linear(120,80)
        self.fc3 = nn.Linear(80,2)
    #pass data to the CNN. x represents the data    
    def forward(self,x):
        x = F.relu(F.avg_pool2d(self.conv1(x),(1,2)))
#        print(x.shape)
        x = F.relu(F.avg_pool2d(self.conv2(x),(1,2)))
 #       print(x.shape)

        x = x.view(x.size(0),-1)
  #      print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# #initialize the network and the optimizer
network = Net()
print(network(example_data).shape)
optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum, weight_decay = L2reg)
#lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader), eta_min = 0)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=10, threshold=1e-4, threshold_mode='rel')

# #store the training results
train_losses = []  #for Q
trainV_losses = [] #for V
train_counter = []
test_losses = [] #for Q
testV_losses = [] #for V
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs+1)]
train_output = [] #for Q
trainV_output = [] #for V
train_target = [] #for Q
trainV_target = [] #for V
test_output = []  
testV_output = []
test_target = []
testV_target = []


# train loop 
def train(epoch):
    
    #lr_scheduler.step()
    
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        output = network(data)
        #Save the training result
        #Q
        train_output.append(output.data[:,0]) 
        train_target.append(target.data[:,0])
        #V
        trainV_output.append(output.data[:,1])
        trainV_target.append(target.data[:,1])
        #MSE loss
        Q_loss = F.mse_loss(output[:,0],target[:,0])
        V_loss = F.mse_loss(output[:,1],target[:,1]) 
        loss = 5*Q_loss + V_loss
        #calculate the gradient
        loss.backward()
        optimizer.step()
        
        
        #gradually print epoch results
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tQloss:{:.6f}\tVloss:{:.6f}\tQ_NN: {:.4f}\tV_NN: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                Q_loss.item(), V_loss.item(), output[-1,0], output[-1,1]))
            #store training results
            train_losses.append(Q_loss.item())
            trainV_losses.append(V_loss.item())
            train_counter.append(
                (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
            
            for param_group in optimizer.param_groups:
                print("Current learning rate: {}".format(param_group['lr']))

            #save the model    
            #torch.save(network.state_dict(), '/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/L3_model.pt')
            #torch.save(optimizer.state_dict(), '/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/L3_optim.pt')

#test loop
def test():
    #global test_output
    network.eval()
    test_loss = 0
    testV_loss = 0
    with torch.no_grad(): #disable the gradient computation
        for data, target in test_loader:
            output = network(data)
            #save the test result
            #Q
            test_output.append(output[:,0])
            test_target.append(target[:,0])
            #V
            testV_output.append(output[:,1])
            testV_target.append(target[:,1])
            #loss
            test_loss += F.mse_loss(output[:,0], target[:,0], size_average=False).item()
            testV_loss += F.mse_loss(output[:,1], target[:,1], size_average=False).item()

    #calculate the average loss per epoch        
    test_loss /= len(test_loader.dataset)
    testV_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    testV_losses.append(testV_loss)
    print('\nTest set: Qloss: {:.6f}, Vloss: {:.6f}, Qn: {:.4f}, Qf: {:.4f}, Vn: {:.4f}, Vf: {:.4f}\n'.format(
                test_loss, testV_loss, output[-1,0], target[-1,0], output[-1,1], target[-1,1]))
                
    #save the model    
    if test_loss <= 0.0075:
        torch.save(network.state_dict(), '/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/NB_model.pt')

    return test_loss

#run the training
print("start training")

test()
for epoch in range(1,n_epochs+1):
    train(epoch)
    Loss = test()
    lr_scheduler.step(Loss)
    

print("finish training")


# Post processing of data

print(len(train_output))
print(len(test_output))

print(len(trainV_output))
print(len(testV_output))

#obtain and convert learning results from list to tensor
train_output = torch.cat(train_output, 0)
train_target = torch.cat(train_target, 0)

trainV_output = torch.cat(trainV_output, 0)
trainV_target = torch.cat(trainV_target, 0)

test_output = torch.cat(test_output,0)
test_target = torch.cat(test_target,0)

testV_output = torch.cat(testV_output,0)
testV_target = torch.cat(testV_target,0)

print(len(train_target))
print(len(test_target))

train_outputArr = train_output.numpy()*0.0922+4.8502e+00  #denormalize for Q
train_targetArr = train_target.numpy()*0.0922+4.8502e+00  #Q

trainV_outputArr = trainV_output.numpy()*0.0002+1.9287e-03  #denormalize for V
trainV_targetArr = trainV_target.numpy()*0.0002+1.9287e-03  #for V

test_outputArr = test_output.numpy()*0.0922+4.8502e+00  #for Q
test_targetArr = test_target.numpy()*0.0922+4.8502e+00  #for Q

testV_outputArr = testV_output.numpy()*0.0002+1.9287e-03  #for V
testV_targetArr = testV_target.numpy()*0.0002+1.9287e-03  #for V

print('Training dataset size: {}'.format(len(trainV_outputArr)))
print('Test dataset size: {}'.format(len(testV_outputArr)))

# %% performance metrics

train_size = 4000
test_size = 1000

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""

   # Compute correlation matrix
    corr_mat = np.corrcoef(x, y)

    #calculate the coeff
    pearson_R = corr_mat[0,1]

   # Return entry [0,1]
    return pearson_R

#calculate the pearson correlation coefficient between output and target values
coeff_train = pearson_r(train_outputArr[len(train_outputArr)-train_size:-1], train_targetArr[len(train_targetArr)-train_size:-1])
coeff_test = pearson_r(test_outputArr[len(test_outputArr)-test_size:-1], test_targetArr[len(test_targetArr)-test_size:-1])
print(train_outputArr[len(train_outputArr)-train_size:-1])
print(train_targetArr[len(train_targetArr)-train_size:-1])
print('Training Q corr coeff: {:.3f}\tTest Q corr coeff: {:.3f}'.format(coeff_train, coeff_test))

coeffV_train = pearson_r(trainV_outputArr[len(trainV_outputArr)-train_size:-1], trainV_targetArr[len(trainV_targetArr)-train_size:-1])
coeffV_test = pearson_r(testV_outputArr[len(testV_outputArr)-test_size:-1], testV_targetArr[len(testV_targetArr)-test_size:-1])
print('Training V corr coeff: {:.3f}\tTest V corr coeff: {:.3f}'.format(coeffV_train, coeffV_test))

#calculate the prediction error between output and target values
def pred_error(x,y):
    """Compute absolute percentage error between two arrays."""

    percent_e = np.absolute((x - y)/y)
    
    return 100*np.mean(percent_e)

predError_train = []
predError_test = []
predErrorV_train = []
predErrorV_test = []

for i in range(n_epochs):

    predError_train.append(pred_error(train_outputArr[i*train_size:(i+1)*train_size-1], train_targetArr[i*train_size:(i+1)*train_size-1]))
    predErrorV_train.append(pred_error(trainV_outputArr[i*train_size:(i+1)*train_size-1], trainV_targetArr[i*train_size:(i+1)*train_size-1]))

    t  = i+1
    predError_test.append(pred_error(test_outputArr[t*test_size:(t+1)*test_size-1], test_targetArr[t*test_size:(t+1)*test_size-1]))
    predErrorV_test.append(pred_error(testV_outputArr[t*test_size:(t+1)*test_size-1], testV_targetArr[t*test_size:(t+1)*test_size-1]))


print('Min pred error train: {:.8f}\nMin pred error test: {:.8f}'.format(min(predError_train), min(predError_test)))
print('Min mse train: {:.8f}\nMin mse test: {:.8f}'.format(min(train_losses), min(test_losses)))

print('V Min pred error train: {:.8f}\nV Min pred error test: {:.8f}'.format(min(predErrorV_train), min(predErrorV_test)))
print('V Min mse train: {:.8f}\nV Min mse test: {:.8f}'.format(min(trainV_losses), min(testV_losses)))


# %% plots

fig = plt.figure()
plt.plot([x / train_size for x in train_counter],train_losses,color='blue')
plt.plot([x / train_size for x in test_counter],test_losses, linewidth = 2,color = 'red')
plt.legend(['Train Loss','Test Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Q Mean Squared Error')
#plt.ylim(-0.015,0.25)
plt.yscale('log')
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/Q_MSE_loss.eps')

fig = plt.figure()
plt.plot([x / train_size for x in train_counter],trainV_losses,color='blue')
plt.plot([x / train_size for x in test_counter],testV_losses, linewidth = 2,color = 'red')
plt.legend(['Train Loss','Test Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('V Mean Squared Error')
#plt.ylim(-0.015,0.25)
plt.yscale('log')
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/V_MSE_loss.eps')

fig = plt.figure()
plt.plot(np.linspace(1,n_epochs, num=n_epochs),predError_train,color='blue')
plt.plot(np.linspace(1,n_epochs, num=n_epochs),predError_test,color = 'red')
plt.legend(['Train error','Test error'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Prediction Error of Q (%)')
plt.yscale('log')
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/Q_pred_error.eps')

fig = plt.figure()
plt.plot(np.linspace(1,n_epochs, num=n_epochs),predErrorV_train,color='blue')
plt.plot(np.linspace(1,n_epochs, num=n_epochs),predErrorV_test,color = 'red')
plt.legend(['Train error','Test error'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Prediction Error of V (%)')
plt.yscale('log')
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/V_pred_error.eps')

# %%

fig = plt.figure()
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.scatter(10**(train_targetArr[len(train_targetArr)-train_size:-1]),10**(train_outputArr[len(train_outputArr)-train_size:-1]), s=5, color = 'purple')
plt.legend(['Training data (5k)\ncorrelation coeff 0.988'], loc='upper right')
x = np.linspace(0,10e4,10)
y = x
plt.plot(x,y,c = 'k')
plt.xlabel('Q_FDTD')
plt.ylabel('Q_NN')
plt.xlim(2e+4,12e+4)
plt.ylim(2e+4,12e+4)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/Qcorr_coeff_train.eps')

fig = plt.figure()
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.scatter(10**(test_targetArr[len(test_targetArr)-test_size:-1]),10**(test_outputArr[len(test_outputArr)-test_size:-1]), s=5, color = 'purple')
plt.legend(['Test data (1k)\ncorrelation coeff 0.987'], loc='upper right')
x = np.linspace(0,10e4,10)
y = x
plt.plot(x,y,c = 'k')
plt.xlabel('Q_FDTD')
plt.ylabel('Q_NN')
plt.axis('square')
plt.xlim(2e+4,12e+4)
plt.ylim(2e+4,12e+4)
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/Qcorr_coeff_test.eps')

fig = plt.figure()
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.scatter((trainV_targetArr[len(trainV_targetArr)-train_size:-1]),(trainV_outputArr[len(trainV_outputArr)-train_size:-1]), s=5, color = 'purple')
plt.legend(['Training data (5k)\ncorrelation coeff 0.844'], loc='upper right')
x = np.linspace(0,0.005,5)
y = x
plt.plot(x,y,c = 'k')
plt.xlabel('V_FDTD')
plt.ylabel('V_NN')
plt.xlim(0.0015,0.003)
plt.ylim(0.0015,0.003)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/Vcorr_coeff_train.eps')

fig = plt.figure()
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.scatter((testV_targetArr[len(testV_targetArr)-test_size:-1]),(testV_outputArr[len(testV_outputArr)-test_size:-1]), s=5, color = 'purple')
plt.legend(['Test data (1k)\ncorrelation coeff 0.805'], loc='upper right')
x = np.linspace(0,0.005,5)
y = x
plt.plot(x,y,c = 'k')
plt.xlabel('V_FDTD')
plt.ylabel('V_NN')
plt.axis('square')
plt.xlim(0.0015,0.003)
plt.ylim(0.0015,0.003)
plt.savefig('/Users/Renjee/Desktop/CUHK/NOEL/Deep learning proj/code/Vcorr_coeff_test.eps')

# %%
