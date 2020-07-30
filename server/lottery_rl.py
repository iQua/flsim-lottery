import os
import logging
import numpy as np
import pickle
import random
import sys
import json
from multiprocessing import Process, Queue

import torch
import torch.multiprocessing as mp
from torchsummary import summary
import torchvision

from server import Server
from client.lth_client import LTHClient # pylint: disable=impoprt-error
import utils.dists as dists  # pylint: disable=no-name-in-module
import utils.fl_model as fl_model 
from utils.load_dataset import get_partition, get_train_set, get_testloader
import open_lth.models.registry as models_registry
from open_lth.cli import runner_registry
from open_lth.pruning.mask import Mask


class RLLotteryServer(LotteryServer):
    """server for open_lth"""

    def __init__(self, config, env=None, agent=None):
        super().__init__(config, env, agent)
    

    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        self.static_global_model_path = self.config.paths.model
        
        # Add fl_model to import path
        sys.path.append(self.static_global_model_path)

        #get server split and clients total indices
        #server: server_indices, clients: label_idx_dict
        self.generate_dataset_splits()
        self.loading = self.config.data.loading
        if self.loading == 'static':
            self.get_clients_splits()
        # Set up simulated server
        self.load_model(self.static_global_model_path)
        self.make_clients()


    def load_model(self, static_global_model_path):

        lottery_runner = runner_registry.get(
            self.config.lottery_args.subcommand).create_from_args( \
                self.config.lottery_args)

        #set up global model
        self.model = models_registry.get(
            lottery_runner.desc.model_hparams, 
            outputs=lottery_runner.desc.train_outputs)

        self.save_model(self.model, static_global_model_path)
    
        self.baseline_weights = fl_model.extract_weights(self.model)

        #extract flattened weights
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, []) 

    #create clients without dataset assigned
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


    def probe(self):
        self.configuration(self.clients)

        proc_queue = Queue()
        
        processes = [mp.Process(target=client.probe, args=(proc_queue,)) \
            for client in self.clients]

        [p.start() for p in processes]
        [p.join() for p in processes]
                
        #get every client path
        all_client_dict = {
            client.client_id: client for client in self.clients}

        while not proc_queue.empty():
            client_id, data_folder, num_samples = proc_queue.get()
            sample_client_dict[client_id].data_folder = data_folder
            sample_client_dict[client_id].report.set_num_samples(num_samples)

        self.testloader = get_testloader(
            self.config.lottery_args.dataset_name, self.server_indices) 
        


        return self.get_global_model(sample_clients)


    #get indices list for a certain label distribution
    def retrieve_indices(self, client_idx, overlap):
        client_indices = []
        for label, idx in self.label_idx_dict.items():
            #already shuffle
            random.shuffle(idx)
            num = client_idx[self.labels.index(label)]
            client_indices.extend(idx[:num])
            if not overlap:
                #delete already retrieved indices
                idx = idx[num:]
        return client_indices
            

    def get_indices(self, dataset,label):
        indices =  []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == label:
                indices.append(i)
        return indices


    def get_label_nums(self, tot_num):
        #get number for each label list 
        client_idx = []
        if self.config.data.IID:
            for label, idx in self.label_idx_dict.items():
                client_idx.insert(self.labels.index(label), int(tot_num/len(self.labels)))
        
        else:
            bias = self.config.data.bias["primary"]
            secondary = self.config.data.bias["secondary"]
            
            pref = random.choice(self.labels)
            majority = int(tot_num * bias)
            minority = tot_num - majority
            #for one client get partition
            client_idx = get_partition(
                self.labels, majority, minority, pref, bias, secondary)

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

        proc_queue = Queue()
        
        processes = [mp.Process(target=client.run, args=(proc_queue,)) \
            for client in sample_clients]

        [p.start() for p in processes]
        [p.join() for p in processes]
                
        #get every client path
        sample_client_dict = {
            client.client_id: client for client in sample_clients}

        while not proc_queue.empty():
            client_id, data_folder, num_samples = proc_queue.get()
            sample_client_dict[client_id].data_folder = data_folder
            sample_client_dict[client_id].report.set_num_samples(num_samples)

        self.testloader = get_testloader(
            self.config.lottery_args.dataset_name, self.server_indices) 
        
        return self.get_global_model(sample_clients)


    def get_pruned_model(self, sample_clients, reports, prune_level):
        # accuracy_dict = { level: global_model_accuracy }
        accuracy_dict = self.__get_accuracy_per_level(sample_clients, reports)
        selected_model_path = os.path.join(self.global_model_path_per_round, \
            'global', f'level_{prune_level}', 'model.pth')
        
        self.model.load_state_dict(torch.load(selected_model_path))
        self.model.eval()

        #get best global model mask and save to the static mask path(update with every round)
        #self.save_global_mask(
            # self.model, self.static_global_model_path+f'/mask.pth')

        # update static global model for next round
        self.save_model(self.model, self.static_global_model_path)
        # backup the seleted global model to round directory
        self.save_model(self.model, self.global_model_path_per_round)

        accuracy = accuracy_dict[prune_level]
        logging.info(f'Selected level-{prune_level} model accuracy: '\
            + '{:.2f}%\n'.format(100 * accuracy))
        
        return accuracy
          

    def train_best_model_rl(self, sample_clients, reports):
        pass
    

    def get_best_model_rl(self, sample_clients, reports):
        pass


    def test_model_accuracy(self, model, updated_weights):
        fl_model.load_weights(model, updated_weights)
        accuracy = fl_model.test(model, self.testloader)

        return model, accuracy 
