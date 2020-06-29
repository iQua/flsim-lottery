import logging
import torch
import torch.nn as nn
import torch.optim as optim
from client import Client

import argparse
import sys

from open_lth.cli import runner_registry
from open_lth.cli import arg_utils
import open_lth.platforms as platforms


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
 
        pass
    
    def set_bias(self, pref, bias):
        pass

    def download(self, argv):
        pass

    def upload(self, argv):
        pass

    def train(self):
        self.platform.run_job(runner_registry.get(
            self.args.subcommand).create_from_args(self.args).run)

    def test(self):
        pass

    def __load_parser(self, parser, json_dict):
        pass

    def configure(self, config):
        # load json config to argparse parser

        parser = argparse.ArgumentParser()
        parser.add_argument('subcommand')
        parser.add_argument('--platform', default='local', \
            help='The platform on which to run the job.')
        parser.add_argument('--display_output_location', action='store_true', \
            help='Display the output location for this job.')

        self.__load_parser(parser, config.lottery)

        platform_name = config["lottery"]["platform"]
        runner_name = config["lottery"]["subcommand"]

        runner_registry.get(runner_name).add_args(parser)
        self.args = parser.parse_args()

        self.platform = \
            platforms.registry.get(platform_name).create_from_args(self.args)

        runner_registry.get(runner_name).add_args(self.args)
            
        
