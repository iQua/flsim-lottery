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

def get_testloader(dataset_name):
    if(dataset_name == 'mnist'):
        testset = datasets.MNIST(root='./data', train=False,
                                       download=True, transform=None)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


#modified from open_lth/training/standard_callbacks
def test(model, testloader):

    images_count = torch.tensor(0.0).to(device)

    total_correct = torch.tensor(0.0).to(device)

    def correct(labels, outputs):
            return torch.sum(torch.eq(labels, output.argmax(dim=1)))
    
    model.eval()
    
    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
        
            total_correct += correct(label, output)

    accuracy = total_correct / images_count
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy