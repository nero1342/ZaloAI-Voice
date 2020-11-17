from torch.utils import data
from PIL import Image
import numpy as np 
import os


class VoiceFolderDataset(data.Dataset):
    def __init__(self, img_dir):
        self.dir = img_dir
        self.label = os.listdir(img_dir)
        self.x = []
        self.y = []
        print(self.label)
        for label in self.label:
            files = os.listdir(os.path.join(img_dir, label))
            for file in files:
                self.x.append(os.path.join(img_dir, label, file))
                self.y.append(int(label)) 
            #print(files)
    def __getitem__(self, index):
        path = self.x[index]
        x = np.load(path)
        x = np.stack((x, x, x), axis = 0)
        y = self.y[index]        
        return x, y

    def __len__(self):
        return len(self.y)

if __name__ == "__main__":
    dataset = VoiceFolderDataset('./../../data/')
    x, y = dataset.__getitem__(0)
    print(x.shape)