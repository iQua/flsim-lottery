import os
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread, get_ident

import torch
import torchvision
from server import Server

# sys.path.append("..")
# sys.path.append("../client/")
#sys.path.append("../client/open_lth/")

from client.lth_client import LTHClient # pylint: disable=impoprt-error
import utils.dists as dists  # pylint: disable=no-name-in-module
import utils.fl_model as fl_model 
from utils.load_dataset import get_partition, get_train_set, get_testloader
import open_lth.models.registry as models_registry
from open_lth.cli import runner_registry

class LotteryServer(Server):
    """server for open_lth"""

    def __init__(self, config):
        super().__init__(config)
    
    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        

        # Add fl_model to import path
        sys.path.append(model_path)

        #arrange data
        self.generate_dataset_index()

        # Set up simulated server
        self.load_model()
        self.make_clients()



    def load_model(self):

        model_path = self.config.paths.model

        lottery_runner = runner_registry.get(
            self.config.lottery_args.subcommand).create_from_args(self.config.lottery_args)
        #set up global model
        self.model = models_registry.get(
            lottery_runner.desc.model_hparams, 
            outputs=lottery_runner.desc.train_outputs)

        self.save_model(self.model, model_path)
        
        self.baseline_weights = fl_model.extract_weights(self.model)

        #extract flattened weights
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, []) 

    def make_clients(self):

        clients = []
        
        for client_id in range(self.config.clients.total):
            
            new_client = LTHClient(client_id, self.config)
            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))
            
        self.clients = clients

    
    def generate_dataset_index(self):
        
        dataset = get_train_set(self.config.lottery_args.dataset_name)

        self.labels = dataset.targets

        if torch.is_tensor(self.labels[0]):
            self.labels = [label.item() for label in self.labels]

        self.labels = list(set(self.labels))
        
        
        self.label_idx_dict = {}
        
        server_split = self.config.data.server_split
        self.server_indices = []
        #get label_idx_dict
        for label in self.labels:

            label_idx = self.get_indices(dataset, label)

            random.shuffle(label_idx)
            
            server_num = int(len(label_idx) * server_split)

            self.server_indices.extend(label_idx[:server_num])

            self.label_idx_dict[self.labels.index(label)] = label_idx[server_num:]

       
        #get id_index_list
        self.loading = self.config.data.loading

        if self.loading == "dynamic":
            self.client_num = self.config.clients.per_round
            
        if self.loading == "static":
            self.client_num = self.config.clients.total

        #get nums for each label for one client partition
        client_idx = self.get_client_idx_list()

        self.id_index_dict = {}
        
        #every client has the same label distribution, only need one to indicate 
        #but need a whole list to get different indexes

        for label, idx in self.label_idx_dict.items():
            #already shuffle
            num_per_client = client_idx[self.labels.index(label)]

            for i in range(self.client_num):


                if i not in self.id_index_dict.keys():
                    self.id_index_dict[i] = []

                beg = num_per_client * i
                end = num_per_client * (i+1)

                self.id_index_dict[i].extend(idx[beg:end])
            

    
    def get_indices(self, dataset,label):
        indices =  []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == label:
                indices.append(i)
        return indices


    def get_client_idx_list(self):

        #get number for each label list (only need one for all clients)
        client_idx = []
        if self.config.data.IID:
            for label, idx in self.label_idx_dict.items():
                client_idx.insert(self.labels.index(label), int(len(idx)/self.client_num))

        else:
            bias = self.config.data.bias["primary"]
            secondary = self.config.data.bias["secondary"]
            
            pref = random.choice(self.labels)
            total_majority = len(self.label_idx_dict[pref])
            majority = int(total_majority / self.client_num)

            #for one client get partition
            client_idx = get_partition(self.labels, majority, pref, bias, secondary)

        return client_idx
        

    @staticmethod  
    def get_dataset_num(dataset_name):
        if dataset_name == "mnist":
            return 60000
        elif dataset_name == "cifar10":
            return 50000
        

    def round(self):
        sample_clients = self.selection()

        self.configuration(sample_clients)

        threads = list()
        for client in sample_clients:
            t = Thread(target=client.run)
            threads.append(t)
            
        for t in threads:
            t.start()
            t.join()

        reports = self.reporting(sample_clients)

        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        fl_model.load_weights(self.model, updated_weights)

        if self.config.paths.reports:
            self.save_reports(round, reports)
        
        self.save_model(self.model, self.config.paths.model)


        testloader = get_testloader(self.config.lottery_args.dataset_name, self.server_indices)  
        accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy




    def configuration(self, sample_clients):

        #for dynamic 
        id_list = list(range(self.config.clients.per_round))


        for client in sample_clients:
            
            if self.loading == "static":
                dataset_indices = self.id_index_dict[client.client_id]
            
            if self.loading == "dynamic":
                i = random.choice(id_list)
                id_list.remove(i)
                dataset_indices = self.id_index_dict[i]

            
            client.set_data_indices(dataset_indices)
            
        if self.config.clients.display_data_distribution:
            self.display_data_distribution(sample_clients[0])

    
    def display_data_distribution(self, client):
        
        dataset_indices = client.dataset_indices

        label_cnt = []
        for i in range(len(self.labels)):
            tot_idx = set(self.label_idx_dict[i])
            intersection = tot_idx.intersection(dataset_indices)
            label_cnt.append(len(intersection))

        tot_num = sum(label_cnt)
        if self.config.data.IID:
            logging.info(f'Total {tot_num} data in one client, {label_cnt[0]} for one label.')

        else:
            pref_num = max(label_cnt)
            bias = round(pref_num / tot_num,2)
            logging.info(f'Total {tot_num} data in one client, label {label_cnt.index(pref_num)} has {bias} of total data.')
            
            