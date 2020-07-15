
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread
import torch
from server import Server

# sys.path.append("..")
# sys.path.append("../client/")
#sys.path.append("../client/open_lth/")

from client.lth_client import LTHClient # pylint: disable=impoprt-error
import utils.dists as dists  # pylint: disable=no-name-in-module
import utils.fl_model as fl_model 
import open_lth.models.registry as models_registry
from open_lth.cli import runner_registry

class LotteryServer(Server):
    """server for open_lth"""

    def __init__(self, config):
        super().__init__(config)
    
    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total

        # Add fl_model to import path
        sys.path.append(model_path)

        # Set up simulated server
        self.generate_dataset_index()
        self.load_model()
        self.make_clients(total_clients)



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

    def make_clients(self, num_clients):
        #todo: 
        #add different loading 

        clients = []

        for client_id in range(num_clients):
            dataset_indices = self.client_id_index_dict[client_id]
            new_client = LTHClient(client_id, dataset_indices)
            clients.append(new_client)
        
        logging.info('Total clients: {}'.format(len(clients)))
        self.clients = clients

        #attach the certain dataset index to each client
        #static loading
    
    
    def generate_dataset_index(self):
        #get client and server dataset
        dataset_name = self.config.lottery_args.dataset_name
        
        total_index = self.get_total_index(dataset_name)
        #generate clients number's different trainset 
        random.shuffle(total_index)

        server_split = int(self.config.data.server_split * len(total_index))
        clients_split = len(total_index) - server_split
        total_clients = total_index[server_split:]
        self.server_testset = total_index[:server_split]
        
        loading = self.config.data.loading
        if loading == 'dynamic':
            client_num = self.config.clients.per_round
            #to finish

        if loading == 'static':
            client_num = self.config.clients.total
            data_num_per_client = int(clients_split / client_num)
            self.client_id_index_dict = {}
            for i in range(client_num):
                beg = data_num_per_client * i 
                end = data_num_per_client * (i+1)
                self.client_id_index_dict[i] = total_clients[beg:end]
        
               
        
    def get_total_index(self, dataset_name):
        if dataset_name == "mnist" or dataset_name == "cifar10":
            num = 60000
        else:
            print("dataset name is wrong")
        return list(range(num))

    def round(self):
        sample_clients = self.selection()

        self.configuration(sample_clients)

        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        reports = self.reporting(sample_clients)

        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        fl_model.load_weights(self.model, updated_weights)

        if self.config.paths.reports:
            self.save_reports(round, reports)
        
        self.save_model(self.model, self.config.paths.model)


        #todo
        #use openlth test to get accuracy 
        testloader = fl_model.get_testloader(self.config.lottery.dataset_name, self.server_testset)  
        accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy




    def configuration(self, sample_clients):

        #todo: check if need to add loading

        for client in sample_clients:
            config = self.config
            client.configure(config)

    
