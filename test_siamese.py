import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tvtf
from tqdm import tqdm

from datasets.siamese_datasets import SiameseVoiceDataset
from utils.getter import get_instance
from utils.device import move_to

import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str,
                    help='path to the csv ')
parser.add_argument('-w', type=str,
                    help='path to weight files')
parser.add_argument('-g', type=int, default=None,
                    help='(single) GPU to use (default: None)')
parser.add_argument('-b', type=int, default=64,
                    help='batch size (default: 64)')
parser.add_argument('-o', type=str, default='test.csv',
                    help='output file (default: test.csv)')
args = parser.parse_args()

# Device
dev_id = 'cuda:{}'.format(args.g) \
    if torch.cuda.is_available() and args.g is not None \
    else 'cpu'
device = torch.device(dev_id)

# Load model
config = torch.load(args.w, map_location=dev_id)
model = get_instance(config['config']['model']).to(device)
model.load_state_dict(config['model_state_dict'])

# Load data
tfs = tvtf.Compose([
    tvtf.Resize((224, 224)),
    tvtf.ToTensor(),
    tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
])

dataset = VoiceImageDataset(args.d)
dataloader = DataLoader(dataset, batch_size=args.b)

label = []
dist = []
with torch.no_grad():
    model.eval()
    for i, (imgs, fns) in enumerate(tqdm(dataloader)):
        # print("Evaluating {}/{}".format(i, len(dataset)))
        imgs = move_to(imgs, device)
        output = model(imgs)
        output1,output2 = output
        dsts = (F.pairwise_distance(output1, output2))
        lbls = dsts <= 1
        for lbl, dst in zip(lbls, dsts):
            label.append(int(lbl))
            dist.append(dst)
        