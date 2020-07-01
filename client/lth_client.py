# pylint: disable=E1101
# pylint: disable=W0312

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from client import Client
from client import Report

import argparse
import sys

from open_lth.cli import runner_registry
from open_lth.cli import arg_utils
import open_lth.platforms.registry as registry



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
        self.platform.run_job(runner_registry.get(
            self.args.subcommand).create_from_args(self.args).run)

        # todo
        weights = ...
        self.report = Report(self)
        self.report.weights = weights
    

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

        self.platform = registry.get(
            config.lottery["platform"]).create_from_args(self.args)
        
