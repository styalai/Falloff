import pygame
import random, time, sys, h5py
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.transform import resize

hf = h5py.File('data.h5', 'r')
frames = torch.tensor(torch.from_numpy(hf["frame"][:]), dtype=torch.float) # (batch_size, 76, 119)
frames =  frames / torch.max(frames)
moves = torch.from_numpy(hf["move"][:])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=(3, 3), stride=1)
        self.fc1 = nn.Linear(24 * 19 * 29, 128)  # Assuming input size is (76, 119) and stride/padding reduce it to (19, 29)
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 output classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Model()
o = model.forward(frames[0].unsqueeze(0))
print(o.shape)
#optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)