
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
batch_size = 64
#if not on google colab, just remove the /content/CSEGroupProject/ portion of each path below
train_dir = 'Fruits/fruits-360_dataset_100x100/fruits-360/Training' #path to be used for ImageFolder
test_dir = 'Fruits/fruits-360_dataset_100x100/fruits-360/Test'
#splitting dataset into training, validation and testing (80% used for training, 20% used for validation, testing file for testing)
def load_split_train_test(): 
    #transform parameters, using normalizations/resizes found in resnet18 documentation
    #many random transofmrations on the training set
    train_transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.RandomResizedCrop(224),    
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_dir,       
                    transform=train_transform)
    test_data = datasets.ImageFolder(test_dir,
                    transform=test_transform)
    #use testloader as validation set later on
    trainLoader = torch.utils.data.DataLoader(train_data,
            batch_size=batch_size, shuffle= True)
    testloader = torch.utils.data.DataLoader(test_data,
                batch_size=batch_size, shuffle= False)
    return trainLoader, testloader
###########Remainder of logic is training/eval logic#############
#printing a bunch of stuff to do with mps (macOS gpu), testing to see if I can use
def gpu_check(device):
    print(torch.mps.is_available())  
    print(torch.mps.current_allocated_memory())
    print(device)
#gpu_check()
def train():
    #function returns in order of train,test, use testing set as validation set here to help with training
    trainloader, valLoader= load_split_train_test()
    #print(trainloader.dataset.classes)
    #Check if GPU is available to use, else use cpu
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(device)
    model = models.resnet18() #defining pretrained resnet model
    num_classes = len(trainloader.dataset.classes)
    #need to define nn as a linear space and give loss functions (cross entropy best for multi-class)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    #Medium article says Momentum is best optimizer for ResNet, trying that now
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum= 0.9) #trying parameters seen in lecture, think lr too high before
    model.to(device)
    epochs = 100 #using 100 epoch for now keep monitoring 
    print("starting training...")
    running_loss = 0
    for epoch in range(epochs):
        #variable for total loss across each epoch
        tloss = 0.0
        #training step
        model.train() 
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            tloss += loss.item()
            if i % batch_size == 0:
                gpu_check(device)
                print(f'[{epoch + 1} loss: {running_loss / batch_size}')
                running_loss = 0.0
        #validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        #CHECK WITH VALIDATOR EVERY 3 EPOCHS
        if (epoch + 1) % 3 == 0:
            with torch.no_grad():
                for inputs, labels in valLoader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    i+=1 
            valAccuracy = 100 * correct / total
            
            print(f'Epoch {epoch+1}, Total average Training Loss: {tloss / len(trainloader)}, Val Loss: {val_loss/(len(valLoader))}, Val Accuracy: {valAccuracy:.2f}%')
    print("reach end, proceeding to save...")
    path = './food.pth'
    torch.save(model.state_dict(),path)
    print("saved model")
#comment in and out to train below
train()
def load_model(model_path, num_classes):
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path), device)
    return model
########### New Test Function ###########
# Function to make a prediction on a single image
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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

model_path = './food.pth'
_, test_loader = load_split_train_test() 
class_names = test_loader.dataset.classes
image_path = 'test_images/greenpear.jpeg'  
saved_model = load_model(model_path, len(class_names))
#predicted_class = predict_image(image_path, saved_model, class_names)
#print(f'Predicted class: {predicted_class}')
def test(model, test_loader):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
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
#test(saved_model, test_loader)