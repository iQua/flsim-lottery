import random
import torch
from torchvision import datasets, transforms

import utils.dists as dists

def get_train_set(dataset_name):
    print(dataset_name)
    if(dataset_name == "mnist"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        dataset = datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
    
    elif(dataset_name == "cifar10"):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32, 4),
            transforms.ToTensor() 
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, 
                                    download=True, transform=transform)

    elif(dataset_name == "fashion_mnist"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        dataset = datasets.FashionMNIST(root='./data', train=True,
                                       download=True, transform=transform)

    else:
        print("Error! dataset name is wrong.")

    return dataset

def get_testloader(dataset_name, indices):

    dataset =  get_train_set(dataset_name)
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset)

    return dataloader

def get_partition(labels, majority, minority, pref, bias, secondary):
    # Get a non-uniform partition with a preference bias

    # Calculate number of minor labels
    len_minor_labels = len(labels) - 1

    if secondary:
        # Distribute to random secondary label
        dist = [0] * len_minor_labels
        dist[random.randint(0, len_minor_labels - 1)] = minority
    else:
        # Distribute among all minority labels
        dist = dists.uniform(minority, len_minor_labels)

    # Add majority data to distribution
    dist.insert(labels.index(pref), majority)

    return dist