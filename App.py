from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import NutritionalDataSet  # Assuming this file contains the dataset


app = Flask(__name__)



batch_size = 64
#if not on google colab, just remove the /content/CSEGroupProject/ portion of each path below
train_dir = 'Fruits/fruits-360_dataset_100x100/fruits-360/Training' #path to be used for ImageFolder
test_dir = 'Fruits/fruits-360_dataset_100x100/fruits-360/Test'


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



transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path, num_classes):
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path), device)
    return model



_, test_loader = load_split_train_test()
class_names = test_loader.dataset.classes
model = load_model('./resnet18_trained.pth', len(class_names))
device = torch.device('cpu')
model.to(device)
model.eval()


@app.route('/classify',methods=['POST'])
def get_image():
    file = request.files['image']
    image= Image.open(file.stream).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
    return jsonify({'class': class_names[class_idx]})

@app.route('/nutrition', methods=['GET'])
def get_nutrition():
    fruit = request.args.get('fruit')
    
    if fruit in NutritionalDataSet.dataset:
        nutrition = NutritionalDataSet.dataset[fruit]
        return jsonify(nutrition)
    else:
        return jsonify({"error": "Fruit not found in dataset"}), 404

@app.route('/')
def get():
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)