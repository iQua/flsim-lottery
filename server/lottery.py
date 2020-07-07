
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread
import torch
from server import Server

sys.path.append("..")
sys.path.append("../client/")
sys.path.append("../client/open_lth/")
from client.lth_client import LTHClient # pylint: disable=impoprt-error
import utils.dists as dists  # pylint: disable=no-name-in-module
import utils.fl_model 
import open_lth.models.registry as models_registry

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
        self.load_model()
        self.make_clients(total_clients)


    def load_model(self):

        model_path = self.config.paths.model
        #set up global model
        self.model = models_registry.get(
            lottery_runner.desc.model_hparams, 
            outputs=lottery_runner.desc.train_outputs)

        self.save_model(self.model, model_path)
        
        self.baseline_weights = extract_weights(self.model)

        #extract flattened weights
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, []) 

    def make_clients(self, num_clients):
        #todo: 
        #add different loading 

        clients = []

        for client_id in range(num_clients):
            new_client = LTHClient(client_id)
            clients.append(new_client)
        
        logging.info('Total clients: {}'.format(len(clients)))
        self.clients = clients
    
    
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
        testloader = fl_model.get_testloader(self.config.lottery.dataset_name)  
        accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy




    def configuration(self, sample_clients):

        #todo: check if need to add loading

        for client in sample_clients:
            config = self.config
            client.configure(config)

    
