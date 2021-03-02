import os
import imageio
import scipy.io
import random
from os import path
from torch.utils.data import Dataset
from torchvision import transforms


class ColonCancerDataset(Dataset):

    CLASSES = [0, 1]

    def __init__(self, directory, seed, train=True):
        cwd = os.getcwd().replace('dataset', '')
        directory = path.join(cwd, directory)

        # randomly split data into training and validation with fixed random seed
        self.data = [os.path.join(directory, x) for x in os.listdir(directory)]
        train_data = random.Random(seed).sample(self.data, 50)

        if train:
            self.data = train_data
            self.image_transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ColorJitter(brightness=0.1,
                                        contrast=0.1),
                 transforms.ToTensor()]
            )
        else:
            self.data = [x for x in self.data if x not in train_data]
            self.image_transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor()]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        folder_path = self.data[idx]
        img_id = int(folder_path.split('/')[-1].replace('img', ''))

        mat = scipy.io.loadmat(
            path.join(folder_path, f"img{img_id}_epithelial.mat"))['detection']

        x = imageio.imread(path.join(folder_path, f"img{img_id}.bmp"))

        x = self.image_transform(x)

        label = int(mat.shape[0] > 0)
        return x, label


if __name__ == '__main__':
    colon_cancer_dataset = ColonCancerDataset('', train=True, seed=13)
    print(colon_cancer_dataset.__getitem__(1))
