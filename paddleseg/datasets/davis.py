import os

from paddleseg.datasets import Dataset
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class DAVIS(Dataset):
    """
    DAVIS 2017 dataset: `https://davischallenge.org/davis2017/code.html`

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory. Default: None
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'trainval', 'trainaug', 'val').
            If you want to set mode to 'trainaug', please make sure the dataset have been augmented. Default: 'train'.
        year (int): DAVIS Challenge dataset year. Default: 2017
        resolution (str): JPEGImgages resolution directory. Default: 480p
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 8

    def __init__(self, transforms, dataset_root=None, mode='train', year=2017, resolution='480p', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.year = year
        self.resolution = resolution
        self.edge = edge

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'val') in DAVIS dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        image_set_dir = os.path.join(self.dataset_root, 'ImageSets', str(self.year))
        if mode == 'train':
            file_path = os.path.join(image_set_dir, 'train.txt')
        elif mode == 'val':
            file_path = os.path.join(image_set_dir, 'val.txt')

        img_dir = os.path.join(self.dataset_root, 'JPEGImages', self.resolution)
        label_dir = os.path.join(self.dataset_root, 'Annotations', self.resolution)

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                image_folder_path = os.path.join(img_dir, line)
                label_folder_path = os.path.join(label_dir, line)

                for image_name, label_name in zip(os.listdir(image_folder_path), os.listdir(label_folder_path)):
                    image_path = os.path.join(image_folder_path, image_name)
                    label_path = os.path.join(label_folder_path, label_name)
                    self.file_list.append([image_path, label_path])
