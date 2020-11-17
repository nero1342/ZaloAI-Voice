from torch.utils import data
from PIL import Image
import numpy as np 
import os
import pandas as pd 
from torchvision import transforms

class VoiceImageDataset(data.Dataset):
    def __init__(self, csv):
        self.csv = csv
        self.df = pd.read_csv(csv)
        self.x = self.df['filename']
        self.y = self.df['category']
        # print(self.df.head())
            #print(files)
    def __getitem__(self, index):
        path = "./../" + self.x[index]
        x = np.asarray(Image.open(path)) / 255.
        #x = np.transpose(x, (2, 0, 1))
        pil_to_tensor = transforms.ToTensor()(x).float()
        # print(pil_to_tensor.shape)
        #x = np.stack((x, x, x), axis = 0)
        y = self.y[index]        
        return pil_to_tensor, y

    def __len__(self):
        return len(self.y)

if __name__ == "__main__":
    dataset = VoiceImageDataset('./../train_train.csv')
    x, y = dataset.__getitem__(0)
    print(x.shape)