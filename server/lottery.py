import os
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread
from multiprocessing import Process

import torch
import json
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

        logging.info('Download datasets if not exist...')
        warmup_client = LTHClient(-1, self.config)
        warmup_client.download_datasets()

        exit(0)
    
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

        processes = [Process(target=client.run) for client in sample_clients]
        [p.start() for p in processes]
        [p.join() for p in processes]
        
        testloader = get_testloader(self.config.lottery_args.dataset_name, self.server_indices) 
        
        #get every client path
        
        client_paths = [client.data_folder for client in sample_clients]

        tot_level = self.config.lottery_args.levels + 1
        ep_num = int(self.config.lottery_args.training_steps[0:-2])

        accuracy_dict = {}
        #for loop
        for i in range(tot_level):
            
            weights = []
            #load path to model 
            for client_path in client_paths:
                path = os.path.join(client_path, f'level_{i}', 'main', 
                        f'model_ep{ep_num}_it0.pth')
                base_model = self.model
                base_model.load_state_dict(torch.load(path))
                base_model.eval()
                weight = fl_model.extract_weights(base_model)
                weights.append(weight)

            #aggregation
            reports = self.reporting(sample_clients)
            updated_weights = self.federated_averaging(reports, weights)

            #test accuracy 
            fl_model.load_weights(base_model, updated_weights)
            accuracy = fl_model.test(base_model, testloader)

            accuracy_dict[i] = accuracy

            model_path = os.path.join(self.global_model_path, 'global')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            test_model_path=model_path+ f'/level_{i}_model.pth'
            torch.save(base_model.state_dict(), test_model_path)        

        with open(os.path.join(self.global_model_path, 'accuracy.json'), 'w') as fp:
            json.dump(accuracy_dict, fp) 

        best_level = max(accuracy_dict, key=accuracy_dict.get)
        best_path = os.path.join(self.global_model_path, 'global', f'level_{best_level}_model.pth')

        self.model.load_state_dict(torch.load(best_path))
        self.model.eval()
        self.save_model(self.model, self.config.paths.model)
        accuracy = accuracy_dict[best_level]
        logging.info('Best average accuracy: {:.2f}%\n'.format(100 * accuracy))

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
            

    def extract_client_updates(self, weights):

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = self.baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def federated_averaging(self, reports, weights):
        
        # Extract updates from reports
        updates = self.extract_client_updates(weights)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)


        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(self.baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights