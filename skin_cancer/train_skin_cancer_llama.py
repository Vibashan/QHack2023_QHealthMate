import os
import sys
import pandas as pd
import argparse
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

# Define the mapping dictionary for class indices
class_mapping = {
    2: 0,
    6: 1,
    1: 2,
    0: 3,
    5: 4,
    3: 5
}

# Define a custom dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, mode='train'):
        self.dataframe = dataframe
        self.data_dir = os.path.join(data_dir, mode)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx, 1]
        img_name = os.path.join(self.data_dir, f'{img_id}.jpg')
        image = Image.open(img_name)
        label = int(self.dataframe.iloc[idx, 10])  # cell_type_idx column

        # Map the original class index to a continuous index using the mapping dictionary
        label = class_mapping[label]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define ResNetSkin model
class ResNetSkin(nn.Module):
    def __init__(self, llama_vector, num_classes=6):
        super(ResNetSkin, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
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
        
def train(model, dataloader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    f1 = f1_score(true_labels, predictions, average='macro')
    cm = confusion_matrix(true_labels, predictions)

    return running_loss / len(dataloader.dataset), f1, cm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Skin Lesion Classification')
    parser.add_argument('--data_dir', type=str, default='../../data/skin_data/', help='Path to the data directory')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory to save the log file')
    parser.add_argument('--model_dir', type=str, default='./models/', help='Directory to save the best model checkpoint')
    parser.add_argument('--llama_vector_path', type=str, default='../../data/skin_data/llama_embed_skin.npy', help='Path to the llama vector file')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Create a log file with a dynamic name based on current date and time
    log_filename = os.path.join(args.log_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # Configure the logging module
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # # Redirect stdout and stderr to the log file
    sys.stdout = open(log_filename, 'a')
    sys.stderr = open(log_filename, 'a')
    
    logging.info("Script started.")

    # Data directory
    train_csv_path = os.path.join(args.data_dir, 'train.csv')
    test_csv_path = os.path.join(args.data_dir, 'test.csv')

    # Read train and test CSV files
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Define transformations for training and validation data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets using the custom SkinLesionDataset class
    train_dataset = SkinLesionDataset(dataframe=df_train, data_dir=args.data_dir, transform=train_transform, mode='train')
    test_dataset = SkinLesionDataset(dataframe=df_test, data_dir=args.data_dir, transform=test_transform, mode='test')

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model with the llama_vector
    llama_vector = np.load(args.llama_vector_path)
    model = ResNetSkin(llama_vector=llama_vector).cuda()

    # For example, the training loop may look like:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    num_epochs = args.num_epochs
    best_f1 = 0.0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        test_loss, test_f1, test_cm = evaluate(model, test_loader, criterion)  
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            model_filename = os.path.join(args.model_dir, f"best_model_final.pth")
            torch.save(model.state_dict(), model_filename)
            logging.info(f"Best model saved to: {model_filename}")

    # Load the best model and evaluate on the test set
    best_model = ResNetSkin(llama_vector=llama_vector).cuda()
    best_model.load_state_dict(torch.load(model_filename))
    test_loss, test_f1, test_cm = evaluate(best_model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")

    # Print classification report and confusion matrix
    print("\nClassification Report:")
    target_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

    # Create empty lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Set the model to evaluation mode
    best_model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass through the model
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)

            # Append true and predicted labels to the lists
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Convert the lists to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Map class indices to their respective names using the class_mapping dictionary
    true_class_names = [target_names[label] for label in true_labels]
    predicted_class_names = [target_names[label] for label in predicted_labels]

    # Print the classification report
    print(classification_report(true_class_names, predicted_class_names, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(test_cm)

    logging.info("Script finished.")


    
