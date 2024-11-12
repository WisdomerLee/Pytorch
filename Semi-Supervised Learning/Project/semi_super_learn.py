
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
from sklearn.metrics import accuracy_score
from PIL import Image
import seaborn as sns

BATCH_SIZE = 10
DEVICE = torch.device("cuda:0" if torch.cuda.isavailable() else "cpu")
NUM_EPOCHS = 50
LOSS_FACTOR_SELFSUPERVISED=0

transform_super = transforms.Compose([
  transforms.Resize(32),
  transforms.Grayscale(num_output_channels=1),
  transforms.ToTensor(),
  transforms.Normalize((0.5, ), (0.5, ))
])


