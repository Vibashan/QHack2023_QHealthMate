import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
import argparse
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

def predict_image_class(input_audio_path, model_path, llama_vector_path):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet-ECG Inference')
    parser.add_argument('--input_audio_path', type=str, required=True, help='Path to the input ECG image')
    parser.add_argument('--llama_vector_path', type=str, default='../../data/ecg_data/llama_embed_ecg.npy', help='Path to the llama vector file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the best model checkpoint')
    args = parser.parse_args()

    # Validate model checkpoint path
    if not os.path.isfile(args.model_path):
        print("Error: Invalid model checkpoint path.")
        exit(1)

    # Validate input image path
    if not os.path.isfile(args.input_audio_path):
        print("Error: Invalid input image path.")
        exit(1)

    # Perform inference on the provided ECG image
    class_name = predict_image_class(args.input_audio_path, args.model_path, args.llama_vector_path)

    print("\n=== Skin Cancer Classification ===")
    print(f"Input Image Path: {args.input_image}")
    print(f"Predicted Skin Lesion Class: {predicted_class_name}")
    print("===============================")
