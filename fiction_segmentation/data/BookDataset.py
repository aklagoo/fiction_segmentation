import numpy as np
import torch
from torch.utils.data import Dataset


class BookDataset(Dataset):
    def __init__(self, datafile, embeddings, win_size, device='cpu'):
        """Loads features and labels from '.npz' file"""
        super(Dataset, self).__init__()
        self.win_size = win_size
        self.device = device

        # Load features and labels
        data = np.load(datafile)

        # Convert to tensors
        self.features = torch.from_numpy(data['features']).to(device)
        self.labels = torch.from_numpy(data['labels']).float().to(device)
        self.embeddings = torch.from_numpy(embeddings).to(device)

    def __getitem__(self, idx):
        """Encodes embeddings, creates batch of left and right context"""
        # Select segments
        _X_left = self.features[idx:idx + self.win_size]
        _X_center = self.features[idx + self.win_size]
        _X_right = self.features[idx + self.win_size + 1:idx + self.win_size * 2 + 1]

        # Encode embeddings
        x_left, x_right = [], []
        for sent in _X_left:
            x_left.append(torch.stack([self.embeddings[idx] for idx in sent]))
        for sent in _X_right:
            x_right.append(torch.stack([self.embeddings[idx] for idx in sent]))
        x_center = [self.embeddings[idx] for idx in _X_center]

        x_left = torch.stack(x_left)
        x_center = torch.stack([torch.stack(x_center)])
        x_right = torch.stack(x_right)

        # Get labels
        y = self.labels[idx + self.win_size]

        return (x_left.to(self.device), x_center.to(self.device), x_right.to(self.device)), y.to(self.device)

    def __len__(self):
        return self.features.shape[0] - 2 * self.win_size
