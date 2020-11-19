import torch
import torch.nn.functional as F

class Accuracy():
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        sample_size = output.size(0)
        return correct, sample_size

    def update(self, value):
        self.correct += value[0]
        self.sample_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        return self.correct / self.sample_size

    def summary(self):
        print(f'Accuracy: {self.value()}')



class AccuracySiamese():
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        output1,output2 = output
        pred = (F.pairwise_distance(output1, output2) <= 1)
        correct = (pred == target).sum()
        sample_size = output1.size(0)
        return correct, sample_size

    def update(self, value):
        self.correct += value[0]
        self.sample_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        return self.correct / self.sample_size

    def summary(self):
        print(f'Accuracy: {self.value()}')
