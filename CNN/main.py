import torch
import torch.nn as nn
import torch.nn.functional as F
import data_loader as dl
import pandas as pd
import data_processor as dp
import os


class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 8 * 32, 128)  # remember to modify the dimension
        self.fc2 = nn.Linear(128, 4)  

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 8 * 32)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        for step, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if step % 100 == 99:
                print('[Epoch: %d, Batch: %5d] Loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
    

def eval_model(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Validation Accuracy: {100 * correct / total}%')

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

def print_model_parameters(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")

def prediction(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            feature_batch = batch.to(device) # ! reminder: batch.size() : torch.Size([32, 1, 64, 259])    batch.size()[0] : torch.Size([1, 64, 259])
            print(feature_batch.size())
            # Add a channel dimension if it's missing
            if feature_batch.dim() == 3:
                 eature_batch = feature_batch.unsqueeze(1) 

            predictions_batch = model(feature_batch)

            _, predicted_classes = torch.max(predictions_batch, 1)
            all_predictions.extend(predicted_classes.cpu().tolist())  # ->cpu ->list

    submission = pd.DataFrame({
        'id': range(len(all_predictions)),  # create a range from 0 to len(all_predictions)-1
        'category': all_predictions  # 使用所有预测类别索引的列表
    })

    submission.to_csv('./submission.csv', index=False)

def main():
    '''
    Assume that you have already processed data_processor.py and got test_images and train_images
    '''

    train_audio = './data/train_mp3s'
    train_image = './data/train_images'
    test_audio = './data/test_mp3s'
    test_image = './data/test_images'
    model_path = "./complete_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(train_image, exist_ok=True)  
    os.makedirs(test_image, exist_ok=True) 
    print("Now begin to process dataset...")
    dp.save_image(train_audio, train_image)
    dp.save_image(test_audio, test_image)

    print("Now begin to load dataset...")
    train_loader, val_loader, test_loader = dl.create_dataloaders('./data/train_images', './data/test_images', './data/train_label.txt')
    model = AudioCNN()
    
    print("Now begin to train model...")
    train_model(model, train_loader, 10)
    save_model(model, model_path)
    #load_model(model, './complete_model.pth')
    #print_model_parameters(model)
    
    print("Now begin to evaluate model...")
    eval_model(model, val_loader)
    
    print("Now begin to predict labels...")
    prediction(model, test_loader)

main()