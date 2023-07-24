import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse

import warnings
warnings.filterwarnings('ignore')

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

class ResNetSkin(nn.Module):
    def __init__(self, llama_vector, num_classes=6):
        super(ResNetSkin, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.llama_vector = nn.Parameter(torch.Tensor(llama_vector), requires_grad=False)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.matmul(x, self.llama_vector)
        return x

def predict_image_class(input_image_path, model_path, llama_vector_path):
    # Load the best model for inference
    llama_vector = np.load(llama_vector_path)
    model = ResNetSkin(llama_vector=llama_vector)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Skin cancer classification model loaded.")

    # Preprocess the input image
    print(f"Processing input image: {input_image_path}")
    input_tensor = preprocess_image(input_image_path)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    # Map the predicted class index to the class name
    class_names = ['Dermatofibroma', 'Benign keratosis-like lesions', 'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma']
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skin Lesion Classification')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image for classification')
    parser.add_argument('--model_path', type=str, default='./models/best_model_final.pth', help='Path to the best model checkpoint file')
    parser.add_argument('--llama_vector_path', type=str, default='../../data/skin_data/llama_embed_skin.npy', help='Path to the llama vector file')
    args = parser.parse_args()

    # Validate input image path
    if not os.path.isfile(args.input_image):
        print("Error: Invalid input image path.")
        exit(1)

    # Validate model checkpoint path
    if not os.path.isfile(args.model_path):
        print("Error: Invalid model checkpoint path.")
        exit(1)

    # Validate llama vector path
    if not os.path.isfile(args.llama_vector_path):
        print("Error: Invalid llama vector path.")
        exit(1)

    predicted_class_name = predict_image_class(args.input_image, args.model_path, args.llama_vector_path)

    print("\n=== Skin Cancer Classification ===")
    print(f"Input Image Path: {args.input_image}")
    print(f"Predicted Skin Lesion Class: {predicted_class_name}")
    print("===============================")
