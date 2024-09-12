"""
todo
不要这个，直接读.npy
"""

import os
import cv2
import numpy as np
import torch
import torch.utils.data as data


class CycleGANDataset(data.Dataset):
    """
    Two data folders,
    for smoking frames, contained 2 image: smoke_0.jpg, pred.jpg;
    for clean frames, contained 2 image: clean_0.jpg, pred.jpg.
    """
    def __init__(self, root_smoke, root_clean, img_size=[256, 320], transform=None):
        super(CycleGANDataset, self).__init__()

        self.img_size = img_size
        self.transform = transform

        self.smoke, self.smoke_p = self.load_data(root_smoke)
        self.clean, self.clean_p = self.load_data(root_clean)

        self.length = min(len(self.smoke), len(self.clean))

    def load_dataset(self, root, type='smoke'):
        """
        :param root: data folder root
        :param type: smoke or clean
        :return: 2 lists of smoke/clean and predict images.
        """
        folders = os.listdir(root)
        image_data = []
        predict_data = []
        for folder in folders:
            if os.path.exists(os.path.join(root, folder, type+'_0.jpg')) and os.path.exists(os.path.join(root, folder, 'pred.jpg')):
                image = cv2.imread(os.path.join(root, folder, type+'_0.jpg'), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
                image_p = cv2.imread(os.path.join(root, folder, 'pred.jpg'), cv2.IMREAD_COLOR)
                image_p = cv2.resize(image_p, (self.img_size[1], self.img_size[0]))
                image_data.append(image)
                predict_data.append(image_p)

        return image_data, predict_data

    def __getitem__(self, index):
        smoke = (torch.tensor(self.smoke[index]).float()/255).permute(2, 0, 1)
        smoke_p = (torch.tensor(self.smoke_p[index]).float()/255).permute(2, 0, 1)
        clean = (torch.tensor(self.clean[index]).float()/255).permute(2, 0, 1)
        clean_p = (torch.tensor(self.clean_p[index]).float()/255).permute(2, 0, 1)

        return smoke, clean, smoke_p, clean_p

    def __len__(self):
        return self.length


def load_data(
        batch_size, val_batch_size,
        root_smoke, root_clean,
        img_size=[256, 320]):

    train_set = CycleGANDataset(root_smoke=root_smoke+'_train', root_clean=root_clean+'_train', img_size=img_size)
    valid_set = CycleGANDataset(root_smoke=root_smoke+'_valid', root_clean=root_clean+'_valid', img_size=img_size)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_valid = torch.utils.data.DataLoader(
        valid_set, batch_size=val_batch_size, shuffle=False, pin_memory=True)

    return dataloader_train, dataloader_valid
