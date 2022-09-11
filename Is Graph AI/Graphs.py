import os
import cv2
from torch.utils.data import Dataset
from torchvision import datasets


class Graphs(Dataset):

    def __init__(self, dir_path, file_list, transform=None):
        self.dir_path = dir_path
        self.file_list = file_list
        self.transform = transform
        self.label_dict = {'just_image': 0, 'chart': 1}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        self.image_name = os.path.join(self.dir_path, self.file_list[idx])
        # print(self.image_name)
        if self.image_name.split('.')[::-1][0] == "gif":
            gif = cv2.VideoCapture(self.image_name)
            _, image = gif.read()
        else:
            image = cv2.imread(self.image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for name, label in self.label_dict.items():
            self.label = ''
            if name in self.image_name.split('/')[::-1][0]:
                self.label = label
                break

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.label


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        # print(path)
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
