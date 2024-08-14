import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 5
batch_size = 10
lr = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
classes = ("airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks")


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        #3 is the number of input channels that are the RGB channels 
        self.pool = nn.MaxPool2d(2,2)
        #kernel size of 2 and stride of 2
        self.conv2 = nn.Conv2d(6,16,5)
        #number of input channels in the next layer must be equal to the number of output channels in the last layer

        #The calculation of the number of inputs to the dense layers must be precalculated using the formula 
        #w2 = ((w1 - f + 2p) / s) + 1 
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        #everything else can vary but the number of inputs going into the dense layers and the outputs of them must be fixed
        #output of the last dense layer is 10 because we have 10 classes. 


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #before passing it to the fully connected dense layers we have to flatten the output of the convpool layers
        x = x.view(-1,16*5*5) # tensor flattened 
        #-1 denotes the number of samples in our batch 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # no softmax activation function needed as it is already handled by the crossentropy loss 
        return x
    

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr)

n_total_steps = len(train_loader)
for epoch in range(num_epoch):
    for i, (images,labels) in enumerate(train_loader):
        #pushing the images and labels to the gpu 
        images = images.to(device)
        labels = labels.to(device)

        #doing a forward pass and computing the loss
        output = model(images)
        loss = criterion(output,labels) 

        #backward propagation after emptying the gradients 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0 :
            print(f'Epoch [{epoch + 1}/{num_epoch}], step [{i+1}/{n_total_steps}], loss :{loss.item():.4f}')
print("Finished Training")


with torch.no_grad():
    #disabling gradient computation 
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

    _,predicted = torch.max(outputs,1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()
    for i in range(batch_size):
        label = labels[i]
        pred = predicted[i]
        if (label == pred):
            n_class_correct[label] += 1 
        n_class_samples[label] += 1 

    acc = 100.0 * n_correct/n_samples 
    print("Accuracy of the network : ",acc)
    for i in range(10):
        if n_class_samples[i] > 0:
            acc = 100*n_class_correct[i]/n_class_samples[i]
            print(f"Accuracy of {classes[i]} : {acc} %")
        else:
            print(f"Accuracy of {classes[i]}: N/A (no samples)")
        
