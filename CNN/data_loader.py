import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from sklearn.model_selection import train_test_split


class PickleAudioDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None): # transform = normalize_mel
        """
        file_paths: List of paths to the pickle files
        labels: Optional, pass labels for supervised learning
        transform: Optional, transformations to apply to the data
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'rb') as f:
            feature = pickle.load(f)
        
        # Transform the feature if a transform is provided
        if self.transform:
            feature = self.transform(feature)
        
        # Convert numpy array to torch.Tensor and add a channel dimension
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # Adds the channel dimension

        if self.labels is not None:
            label = self.labels[idx]
            return feature, label
        
        return feature
    
def normalize_mel(mel_spec):
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
    return mel_spec

def load_labels(label_file):
    with open(label_file, "r") as f:
        return [int(label.strip()) for label in f]

def load_dataset_paths(root_dir, file_extension='.pkl'):
    return [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(file_extension)]

def create_dataloaders(train_dir, test_dir, label_file, batch_size=32, test_batch_size=32, val_size=0.2, shuffle=True):
    images_paths = load_dataset_paths(train_dir)
    labels = load_labels(label_file)
    # print(len(images_paths))  11886

    train_paths, val_paths, train_labels, val_labels = train_test_split(images_paths, labels, test_size=val_size, random_state=42)

    train_dataset = PickleAudioDataset(train_paths, train_labels)
    val_dataset = PickleAudioDataset(val_paths, val_labels)
    test_paths = load_dataset_paths(test_dir)
    test_dataset = PickleAudioDataset(test_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# create DataLoader
# train_loader, val_loader, test_loader = create_dataloaders('./data/train_images', './data/test_images', './data/train_label.txt')

'''
for features, labels in train_loader:
    print(features.shape)  # torch.Size([32, 1, 64, 259])
    print(labels.shape)   # torch.Size([32])

'''