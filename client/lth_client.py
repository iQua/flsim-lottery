# pylint: disable=E1101
# pylint: disable=W0312
import sys
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("./client/open_lth/")

from client.client import Client, Report
from utils.fl_model import extract_weights
from open_lth.cli import runner_registry
from open_lth.cli import arg_utils

import open_lth.models.registry as models_registry
import open_lth.platforms.registry as platforms_registry




class LTHClient(Client):
    """Federated learning client enabled with Lottery Ticket."""

    def __init__(self, client_id, config):
        """
        Initialize open_lth
        """
        super().__init__(client_id)
        self.client_id = client_id
        self.args = config.lottery_args

        

    def __repr__(self):
        return 'LTH-Client #{}: {} samples'.format(
            self.client_id, len(self.dataset_indices))

    def set_data_indices(self, dataset_indices):
        
        self.dataset_indices = dataset_indices
    
    def set_task(self, task):
        self.task = task

    def set_mode(self, mode):
        self.mode = mode

    def train(self):
        
        logging.info(f'training on client {self.client_id}')

        self.configure()
        
            #get lotteryRunner
            
        lottery_runner = runner_registry.get(
            self.args.subcommand).create_from_args(self.args)
        
            #run lottery
        self.platform.run_job(lottery_runner.run)
        
        total_levels = self.args.levels
        target_level = total_levels

        epoch_num = int(self.args.training_steps[0:-2])
        
        lottery_folder = lottery_runner.desc.lottery_saved_folder
        path_to_lottery = os.path.join(lottery_folder, 
                        f'replicate_{lottery_runner.replicate}', 
                        f'level_{target_level}', 'main', 
                        f'model_ep{epoch_num}_it0.pth')
        
        


        #init the model
        self.model = models_registry.get(
            lottery_runner.desc.model_hparams, 
            outputs=lottery_runner.desc.train_outputs)

        #load lottery
        self.model.load_state_dict(torch.load(path_to_lottery))
        weights = extract_weights(self.model)

        self.report = Report(self)
        #set dataset number 
        self.report.set_num_samples(len(self.dataset_indices))
        self.report.weights = weights
    

    def test(self):
        pass

    def configure(self):
        """
        config: config object load from json
        """        
        self.args.client_id = self.client_id
        self.args.index_list = ' '.join([str(index) for index in self.dataset_indices])



        self.platform = platforms_registry.get(self.args.platform).create_from_args(self.args)

        


        
    
        
