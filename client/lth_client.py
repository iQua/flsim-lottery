# pylint: disable=E1101
# pylint: disable=W0312
import sys
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("open_lth/")

from client.client import Client, Report
from utils.fl_model import extract_weights
from open_lth.cli import runner_registry
from open_lth.cli import arg_utils

import open_lth.models.registry as models_registry
import open_lth.platforms.registry as platforms_registry





class LTHClient(Client):
    """Federated learning client enabled with Lottery Ticket."""

    def __init__(self, client_id):
        """
        Initialize open_lth
        """
        super().__init__(client_id)
        

    def __repr__(self):
        return 'LTH-Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data), 
            set([label for _, label in self.data]))

    def set_data(self, data, config):
        """
        Set data in open_lth
        """
        dataset_name = self.args.dataset_name

        pass
    
    def set_bias(self, pref, bias):
        pass

    def train(self):
        #todo:
        #add set_data and set_model part in lotteryRunner
        #delete the bias and pref part in lottery config
        #no need to init the model with the global model, not by the config
        #get lotteryRunner
        lottery_runner = runner_registry.get(
            self.args.subcommand).create_from_args(self.args)
        
        #run lottery
        self.platform.run_job(lottery_runner.run)
            
        target_level = 1
        epoch_num = int(self.args.training_steps[0])
        print(epoch_num)

        lottery_folder = lottery_runner.desc.lottery_saved_folder
        path_to_lottery = os.path.join(lottery_folder, 
                        f'replicate_{lottery_runner.replicate}', 
                        f'level_{target_level}', 'main', 
                        f'model_ep{epoch_num}_it0.pth')
        print(path_to_lottery)

        #init the model
        self.model = models_registry.get(
            lottery_runner.desc.model_hparams, 
            outputs=lottery_runner.desc.train_outputs)

        #load lottery
        self.model.load_state_dict(torch.load(path_to_lottery))
        weights = extract_weights(self.model)

        self.report = Report(self)
        self.report.weights = weights
    

    def test(self):
        pass

    def configure(self, config):
        """
        config: config object load from json
        """        

        self.args = config.lottery_args

        self.args.client_id = self.client_id

        self.platform = platforms_registry.get(
            config.lottery["platform"]).create_from_args(self.args)

        print(self.args)

        #download most recent global model
        path = config.paths.model + '/global'

        
    
        
