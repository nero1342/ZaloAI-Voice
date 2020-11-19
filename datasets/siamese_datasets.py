import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms


class SiameseVoiceDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, csv, train):
        self.csv = csv
        self.df = pd.read_csv(csv)
        self.train = train 

        if self.train:
            self.train_labels = self.df['category']
            self.train_data = self.df['filename']
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.df['category']
            self.test_data = self.df['filename']
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        # print(img1, img2)
        img_1 = np.asarray(Image.open("./../" + img1)) / 255.
        img_2 = np.asarray(Image.open("./../" + img2)) / 255.
        img11 = transforms.ToTensor()(img_1).float()
        img22 = transforms.ToTensor()(img_2).float()
        #img1 = Image.fromarray(img1, mode='L')
        #img2 = Image.fromarray(img2, mode='L')
        # if self.transform is not None:
        #     img1 = self.transform(img1)
        #     img2 = self.transform(img2)
        # print(img1, img2, target)
        return (img11, img22), target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        return len(self.test_pairs)


if __name__ == "__main__":
    dataset = SiameseVoiceDataset('./../../train_train.csv', True)
    for i in range(10):
        x, y = dataset.__getitem__(i)
        print(x[0].shape)