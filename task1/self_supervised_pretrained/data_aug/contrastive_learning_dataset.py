import os
from torchvision.transforms import transforms
from .gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from .view_generator import ContrastiveLearningViewGenerator
from ..exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, mean, std, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        return data_transforms

    def get_dataset(self, name, n_views, mean, std):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32, mean, std),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96, mean, std),
                                                              n_views),
                                                          download=True),
                          
                          'imagenet': lambda: datasets.ImageNet(os.path.join(self.root_folder, 'imagenet'),
                                                                transform=ContrastiveLearningViewGenerator(
                                                                     self.get_simclr_pipeline_transform(224, mean, std),
                                                                     n_views)
                                                                )}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
