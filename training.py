
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from PIL import Image
batch_size = 16
train_dir = 'Fruits/fruits-360_dataset_100x100/fruits-360/Training' #path to be used for ImageFolder
test_dir = 'Fruits/fruits-360_dataset_100x100/fruits-360/Test'
#splitting dataset into training and testing (80% used for training, rest for testing)
def load_split_train_test(): 
    #transform parameters, using normalizations/resizes found in resnet18 documentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),  #random crop for more variability,increase accuracy
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #color transforms to increase accuracy further
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_dir,       
                    transform=transform)
    test_data = datasets.ImageFolder(test_dir,
                    transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data,
                   batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data,
                   batch_size=batch_size)
    return trainloader, testloader
###########Remainder of logic is training logic#############
def train():
    trainloader, _ = load_split_train_test()
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
    epochs = 5 #using 4 epoch for now, may increase time permitting (10 too much, 1 too little)
    steps = 0
    running_loss = 0
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
            if i % batch_size == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_size}')
                running_loss = 0.0
    print("reach end, proceeding to save...")
    path = './food.pth'
    torch.save(model.state_dict(),path)
    print("saved model")
#comment in and out to train below
#train()
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path), device)
    model.eval()  # Set to evaluation mode
    return model
########### New Test Function ###########
# Function to make a prediction on a single image
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),  #random crop for more variability,increase accuracy
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #color transforms to increase accuracy further
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
    return class_names[class_idx]

model_path = './food.pth'
_, test_loader = load_split_train_test() 
class_names = test_loader.dataset.classes
image_path = 'test_images/blueberry.jpeg'
model = load_model(model_path, len(class_names))
predicted_class = predict_image(image_path, model, class_names)
print(f'Predicted class: {predicted_class}')
#comment in and out to test
#test()