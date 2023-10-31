import torch
import torchvision.transforms as transforms

def imgnet_transform(is_training=True):
    if is_training:
        transform_list = transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.5,
                                                                    contrast=0.5,
                                                                    saturation=0.3),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    else:
        transform_list = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    return transform_list


def get_imagenet_loader(data_route,batch_size,num_workers):
    train_set = torchvision.datasets.ImageNet(root=root, split="train", download=False,
                                                  transform=imgnet_transform(True))
    test_set = torchvision.datasets.ImageNet(root=root, split="val", download=False,
                                                 transform=imgnet_transform(False))