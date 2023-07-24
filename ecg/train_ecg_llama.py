import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils import resample
import logging
import warnings
warnings.filterwarnings('ignore')

# Define functions and classes

def parse_arguments():
    parser = argparse.ArgumentParser(description='ResNet-ECG Training and Evaluation')
    parser.add_argument('--data_path', type=str, default='../../data/ecg_data/', help='Path to the directory containing data files')
    parser.add_argument('--llama_vector_path', type=str, default='../../data/ecg_data/llama_embed_ecg.npy', help='Path to the llama vector file')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory to save the log file')
    parser.add_argument('--model_dir', type=str, default='./models/', help='Directory to save the best model checkpoint')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    return parser.parse_args()

def preprocess_data(train_df):
    # Upsample minority classes and downsample the majority class to balance the dataset
    df_0 = train_df[train_df[187] == 0].sample(n=20000, random_state=42)
    df_1 = train_df[train_df[187] == 1]
    df_2 = train_df[train_df[187] == 2]
    df_3 = train_df[train_df[187] == 3]
    df_4 = train_df[train_df[187] == 4]
    df_1_upsample = resample(df_1, replace=True, n_samples=20000, random_state=123)
    df_2_upsample = resample(df_2, replace=True, n_samples=20000, random_state=124)
    df_3_upsample = resample(df_3, replace=True, n_samples=20000, random_state=125)
    df_4_upsample = resample(df_4, replace=True, n_samples=20000, random_state=126)
    train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])
    target_train = train_df[187]
    y_train = torch.tensor(target_train.values).long()
    X_train = torch.tensor(train_df.iloc[:, :186].values).float().unsqueeze(1)
    return X_train, y_train

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


'''
python script_name.py --data_path path/to/data/ --llama_vector_path path/to/llama_vector.npy --log_dir path/to/log/directory/ --model_dir path/to/model/directory/ --num_epochs num_epochs --batch_size batch_size --learning_rate learning_rate
'''
if __name__ == '__main__':
    args = parse_arguments()

    # Create a log directory if it doesn't exist
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

    # Log the start of the script
    logging.info("Script started.")

    # Load data
    train_df = pd.read_csv(os.path.join(args.data_path, 'mitbih_train.csv'), header=None)
    test_df = pd.read_csv(os.path.join(args.data_path, 'mitbih_test.csv'), header=None)

    # Preprocess data
    train_df[187] = train_df[187].astype(int)
    X_train, y_train = preprocess_data(train_df)
    X_test = torch.tensor(test_df.iloc[:, :186].values).float().unsqueeze(1)
    y_test = torch.tensor(test_df[187].values).long()

    # Resize the ECG data
    X_train_resized = resize_ecg_data(X_train)
    X_test_resized = resize_ecg_data(X_test)

    # Make sure target tensors have the same number of samples as input tensors
    y_train_resized = y_train[:len(X_train_resized)]
    y_test_resized = y_test[:len(X_test_resized)]

    # Data loaders
    train_dataset = TensorDataset(X_train_resized, y_train_resized)
    test_dataset = TensorDataset(X_test_resized, y_test_resized)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Generate a fixed llama vector using numpy
    llama_vector = np.load(args.llama_vector_path)

    # Create model and move to device
    model = ResNetECG(llama_vector=llama_vector).cuda()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_f1 = 0.0
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        test_loss, test_f1, test_cm = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            # Save the best model checkpoint
            model_filename = os.path.join(args.model_dir, f"best_model_best.pth")
            torch.save(model.state_dict(), model_filename)
            logging.info(f"Best model saved to: {model_filename}")

    # Load the best model and evaluate on the test set
    best_model = ResNetECG(llama_vector=llama_vector).cuda()
    best_model.load_state_dict(torch.load(model_filename))
    test_loss, test_f1, test_cm = evaluate(best_model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")

    # Move the input tensor to the GPU
    X_test_resized = X_test_resized.cuda()

    # Create an empty list to store the predictions
    predictions = []

    # Set the model to evaluation mode
    best_model.eval()

    # Iterate over smaller batches of the test data
    with torch.no_grad():
        for i in range(0, len(X_test_resized), len(X_test_resized)//10):
            inputs = X_test_resized[i:i+len(X_test_resized)//10]
            outputs = best_model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)

    # Convert the list of predictions to a numpy array
    predictions = np.array(predictions)

    # Print the classification report
    print("\nClassification Report:")
    target_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    print(classification_report(y_test_resized.cpu().numpy(), predictions, target_names=target_names))

    print("\nConfusion Matrix:")
    print(test_cm)

    # Log the end of the script
    logging.info("Script finished.")

