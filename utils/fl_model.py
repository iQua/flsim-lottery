import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')

def extract_weights(model):
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if(weight.requires_grad):
            weights.append((name, weight.data))
    return weights

def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)

def get_testloader(dataset_name, indices):
    if(dataset_name == 'mnist'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        dataset = datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)

    if(dataset_name == 'cifar10'):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32, 4),
            transfroms.ToTensor()
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, 
                                    download=True, transform=transform)


    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset)

    return dataloader

#modified from open_lth/training/standard_callbacks
def test(model, testloader):

    model.to(device)
    model.eval()
    
    correct = 0
    total = len(testloader.dataset)

    with torch.no_grad():
        for image, label in testloader:

            image, label = image.to(device), label.to(device)
            output = model(image)
            
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy