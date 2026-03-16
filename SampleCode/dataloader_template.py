import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file, split='train'):
        # Load the data
        # split the data into train/val/test
        

    def __len__(self):
        # return the number of data points

    def __getitem__(self, idx):
        # return an item and its label at 'idx'