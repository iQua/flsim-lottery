import random
import torch
import os
import numpy as np
import pandas as pd
import csv
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import gdown
import zipfile
import PIL
from open_lth.datasets.celeba import CelebaDataset
import utils.dists as dists

def get_train_set(dataset_name):
    if(dataset_name == "mnist"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        dataset = datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
        

    if(dataset_name == "cifar10"):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32, 4),
            transforms.ToTensor() 
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, 
                                    download=True, transform=transform)

    if(dataset_name == "celeba"):
        process_celeba_dataset()
        csv_path = '/mnt/open_lth_datasets/CelebA/data/all_data/all_data.csv'
        root_path = '/mnt/open_lth_datasets/CelebA/data/img_align_celeba/img_align_celeba'
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(84),
            transforms.ToTensor()
        ])
        dataset = CelebaDataset(csv_path, root_path, transform=transform)

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

#this function to read the celebA data and process it then save
# adpated from leaf/data/celeba/metadata_to_json.py
TARGET_NAME = 'Smiling'
parent_path = '/mnt/open_lth_datasets/CelebA'

#pylint-disable= syntax-error
def process_celeba_dataset():
    if os.path.exists('/mnt/open_lth_datasets/CelebA/data/all_data/all_data.csv'):
        print('Datasets already processed')
        return
    download_celeba_dataset() 
    extract_images()  
    identities, attributes = get_metadata()
    celebrities = get_celebrities_and_images(identities)
    targets = get_celebrities_and_target(celebrities, attributes)
    csv_data = build_csv_format(celebrities, targets)
    write_csv(csv_data)


# download  ```identity_CelebA.txt``` and ```list_attr_celeba.txt```
# place them inside the ```data/raw``` folder.
#Download the celebrity faces dataset from the same site. 
# Place the images in a folder named ```img_align_celeba``` 
def download_celeba_dataset():
    if os.path.exists('/mnt/open_lth_datasets/CelebA/data/raw'):
        print('Raw datasets already downloaded and exists')
        return
    file_folder = os.path.join(parent_path, 'data', 'raw')
    os.makedirs(file_folder, exist_ok=True)
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt")
    ]
    for fid, _, fname in file_list:

        url = 'https://drive.google.com/uc?id='+fid
        output = os.path.join(file_folder, fname)
        gdown.download(url, output, quiet=False)

def extract_images():
    img_folder = os.path.join(parent_path, 'data', 'img_align_celeba')
    if os.path.exists(img_folder):
        print('Images already extracted.')
        return
    with zipfile.ZipFile(os.path.join(parent_path, 'data', 'raw', 'img_align_celeba.zip'), "r") as f:
            f.extractall(img_folder)

def get_metadata():
    f_identities = open(os.path.join(
        parent_path, 'data', 'raw', 'identity_CelebA.txt'), 'r')
    identities = f_identities.read().split('\n')

    f_attributes = open(os.path.join(
        parent_path, 'data', 'raw', 'list_attr_celeba.txt'), 'r')
    attributes = f_attributes.read().split('\n')

    return identities, attributes


def get_celebrities_and_images(identities):
    all_celebs = {}

    for line in identities:
        info = line.split()
        if len(info) < 2:
            continue
        image, celeb = info[0], info[1]
        if celeb not in all_celebs:
            all_celebs[celeb] = []
        all_celebs[celeb].append(image)

    good_celebs = {c: all_celebs[c] for c in all_celebs if len(all_celebs[c]) >= 5}
    return good_celebs


def _get_celebrities_by_image(identities):
    good_images = {}
    for c in identities:
        images = identities[c]
        for img in images:
            good_images[img] = c
    return good_images


def get_celebrities_and_target(celebrities, attributes, attribute_name=TARGET_NAME):
    col_names = attributes[1]
    col_idx = col_names.split().index(attribute_name)

    celeb_attributes = {}
    good_images = _get_celebrities_by_image(celebrities)

    for line in attributes[2:]:
        info = line.split()
        if len(info) == 0:
            continue

        image = info[0]
        if image not in good_images:
            continue
        
        celeb = good_images[image]
        att = (int(info[1:][col_idx]) + 1) / 2
        
        if celeb not in celeb_attributes:
            celeb_attributes[celeb] = []

        celeb_attributes[celeb].append(att)

    return celeb_attributes


def build_csv_format(celebrities, targets):
    all_data = []
    tot_num = 0
    for c in celebrities.keys():
        tot_num += len(celebrities[c])
        for i in range(len(celebrities[c])):
            all_data.append((celebrities[c][i], targets[c][i]))
    
    print(f'total num in celeba dataset is {tot_num}')
    return all_data


def write_csv(csv_data):
    file_name = 'all_data.csv'
    dir_path = os.path.join(parent_path, 'data', 'all_data')

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    file_path = os.path.join(dir_path, file_name)

    print('writing {}'.format(file_name))
    with open(file_path, 'w') as outfile:
        csv_out = csv.writer(outfile)
        #'name.jpg', 'target'
        csv_out.writerow(['img_name', 'target'])
        for row in csv_data:
            csv_out.writerow(row)




        