# pylint: disable=E1101
# pylint: disable=W0312

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from client import Client
from client import Report

import os
import argparse
import sys

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
        # todo
        # get lotteryRunner
        lottery_runner = runner_registry.get(self.args.subcommand).create_from_args(self.args)
        # model saved location: lotteryRunner.desc.run_path(lotteryRunner.replicate, level)
        self.platform.run_job(lottery_runner.run)
        
        #path_to_lottery = "/home/ubuntu/open_lth_data/lottery_9e01d1d4915e54abaeca655f83b1106e/replicate_1/level_0/main"
        
        target_level = 1
        epoch_num = int(self.args.training_steps[0])
        print(epoch_num)

        lottery_folder = lottery_runner.desc.lottery_saved_folder
        path_to_lottery = os.path.join(lottery_folder, f'replicate_{lottery_runner.replicate}', 
                                            f'level_{target_level}', 'main', f'model_ep{epoch_num}_it0.pth')
        print(path_to_lottery)

        #init the model
        self.model = models_registry.get(lottery_runner.desc.model_hparams, outputs=lottery_runner.desc.train_outputs)
        #load lottery
        self.model.load_state_dict(torch.load(path_to_lottery))
        weights = self.extract_weights(self.model)

        self.report = Report(self)
        self.report.weights = weights
    
    def extract_weights(self, model):
        weights = []
        for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
            if(weight.requires_grad):
                weights.append((name, weight.data))
        return weights

    def test(self):
        pass

    def configure(self, config):
        """
        config: config object load from json
        """        
        def load_parser(json_dict):
            t_args = argparse.Namespace()
            t_args.__dict__.update(json_dict)
            return t_args
            
        # load arguments from config
        self.args = load_parser(config.lottery) 
        
        self.platform = platforms_registry.get(
            config.lottery["platform"]).create_from_args(self.args)

    
        
