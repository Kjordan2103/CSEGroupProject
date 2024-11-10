
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from PIL import Image
batch_size = 64
#if not on google colab, just remove the /content/CSEGroupProject portion of each path below
train_dir = '/content/CSEGroupProject/Fruits/fruits-360_dataset_100x100/fruits-360/Training' #path to be used for ImageFolder
test_dir = '/content/CSEGroupProject/Fruits/fruits-360_dataset_100x100/fruits-360/Test'
#splitting dataset into training, validation and testing (80% used for training, 20% used for validation, testing file for testing)
def load_split_train_test_val(): 
    #transform parameters, using normalizations/resizes found in resnet18 documentation
    transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.RandomResizedCrop(224),    
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #color transforms to increase accuracy further
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_dir,       
                    transform=transform)
    test_data = datasets.ImageFolder(test_dir,
                    transform=transform)
    #need to split train data into validation set, want 80-20 split
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(num_train * 0.2))
    np.random.shuffle(indices)
    trainIdx, valIndx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(trainIdx)
    val_sampler = SubsetRandomSampler(valIndx)
    trainLoader = torch.utils.data.DataLoader(train_data,
                sampler=train_sampler, batch_size=batch_size)
    valLoader = torch.utils.data.DataLoader(train_data,
                sampler = val_sampler, batch_size= batch_size)
    testloader = torch.utils.data.DataLoader(test_data,
                batch_size=batch_size, shuffle= False)
    return trainLoader, valLoader, testloader
###########Remainder of logic is training logic#############
def gpu_check():
    print(torch.cuda.is_available())  
    print(torch.cuda.current_device()) 
    print(torch.cuda.get_device_name(0)) 
def train():
    #function returns in order of train, val, test
    trainloader, valLoader, _ = load_split_train_test_val()
    print(trainloader.dataset.classes)
    #Check if GPU is available to use, else use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True) #defining pretrained resnet model
    num_classes = len(trainloader.dataset.classes)
    #need to define nn as a linear space and give loss functions (cross entropy best for multi-class)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    #Medium article says Momentum is best optimizer for ResNet, trying that now
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum= 0.9) #trying parameters seen in lecture, think lr too high before
    model.to(device)
    epochs = 17 #using 17 epoch for now keep monitoring 
    running_loss = 0
    for epoch in range(epochs):
        #training step
        model.train() 
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valLoader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        valAccuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Train Loss: {running_loss/batch_size}, Val Loss: {val_loss/batch_size}, Val Accuracy: {valAccuracy:.2f}%')
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
        transforms.RandomResizedCrop(224),    
        transforms.RandomRotation(8), #same idea
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #color transforms to increase accuracy further
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
"""def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
    return class_names[class_idx]"""

model_path = '/content/CSEGroupProject/food.pth'
_, _, test_loader = load_split_train_test_val() 
class_names = test_loader.dataset.classes
image_path = '/content/CSEGroupProject/test_images/redapple.jpeg'  
saved_model = load_model(model_path, len(class_names))
#predicted_class = predict_image(image_path, saved_model, class_names)
#print(f'Predicted class: {predicted_class}')
def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
#comment in and out to test
#test(model, test_loader)