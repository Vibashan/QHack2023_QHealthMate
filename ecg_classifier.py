import os
import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
from PIL import Image

def resize_ecg_data(data):
    # Resize the ECG data to match ResNet-18 input shape (3, 128, 128)
    N, C, d1 = data.shape
    data = data.repeat(1, 3, 1)
    data = data.view(-1, 3, 31, 6)
    data = data.contiguous()
    resized_data = torch.nn.functional.interpolate(data, size=(128, 128), mode='bicubic', align_corners=False)
    return resized_data

class ResNetECG(nn.Module):
    def __init__(self, llama_vector, num_classes=5):
        super(ResNetECG, self).__init__()
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

def predict_ecg_class(input_audio_path, model_path, llama_vector_path):
    # Load the fixed llama vector using numpy
    llama_vector = np.load(llama_vector_path)

    # Create the model and load the best weights
    model = ResNetECG(llama_vector=llama_vector).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the input image
    with Image.open(input_audio_path) as img:
        img = img.convert('L')  # Convert to grayscale
        ecg_signal = np.array(img)
        ecg_signal = torch.tensor(ecg_signal).float().unsqueeze(0).unsqueeze(1)
        resized_ecg_signal = resize_ecg_data(ecg_signal)

    # Move the data to the GPU if available
    if torch.cuda.is_available():
        resized_ecg_signal = resized_ecg_signal.cuda()

    # Perform inference
    with torch.no_grad():
        outputs = model(resized_ecg_signal)

    # Get the predicted class index
    _, predicted_class = torch.max(outputs, 1)

    # Get the class name from the predicted class index
    class_names = ['Non-ecotic beats', 'Supraventricular ectopic beats', 'Ventricular ectopic beats', 'Fusion Beats', 'Unknown Beats']
    predicted_class_name = class_names[predicted_class.item()]

    return predicted_class_name