
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
batch_size = 64
data_dir = 'Fruits/fruits-360_dataset_100x100/fruits-360/Training' #path to be used for ImageFolder
#splitting dataset into training and testing (80% used for training, rest for testing)
def load_split_train_test(datadir, valid_size = .2 ): 
    #transform parameters, using normalizations/resizes found in resnet18 documentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=transform)
    test_data = datasets.ImageFolder(datadir,
                    transform=transform)
    num_train = len(train_data)
    indices = list(range(num_train)) #length of dataset
    #compute point to split with 80-20 split (based off valid_size paramater)
    split = int(np.floor(valid_size * num_train)) 
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    #Use random sampler to split datasets into random images of respective sizes
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=batch_size)
    return trainloader, testloader
###########Remainder of logic is training logic#############
trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)
#Check if GPU is available to use, else use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True) #defining pretrained resnet model
print(model)
num_classes = len(trainloader.dataset.classes)
#need to define nn as a linear space and give loss functions (cross entropy best for multi-class)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
criterion = torch.nn.CrossEntropyLoss()
#Medium article says Momentum is best optimizer for ResNet, trying that now
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum= 0.9)
model.to(device)
epochs = 1 #using 1 epoch for now, may increase time permitting
steps = 0
running_loss = 0
train_losses, test_losses = [], []
for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(trainloader,0):
        inputs, labels = inputs.to(device), labels.to(device)
        #zero paramater gradients
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print("reach end, proceeding to save...")
path = './food.pth'
torch.save(model,path)
print("saved model")








