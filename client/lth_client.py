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
import open_lth.datasets.registry as datasets_registry
import open_lth.platforms as platforms

class LTHClient(Client):
    """Federated learning client enabled with Lottery Ticket."""

    def __init__(self, client_id, config):
        """
        Initialize open_lth
        """
        super().__init__(client_id)
        
        self.args = config.lottery_args
        self.dataset_indices = []
        

    def __repr__(self):
        return 'LTH-Client #{}: {} samples'.format(
            self.client_id, len(self.dataset_indices))

    def set_data_indices(self, dataset_indices):
        
        self.dataset_indices = dataset_indices
    

    def download_datasets(self):
        self.configure()
        
        lth_runner = runner_registry.get(
            self.args.subcommand).create_from_args(self.args)
        
        #run lottery
        # self.platform.run_job(lth_runner.run)
        platforms.platform._PLATFORM = self.platform
        dataset_hparams = lth_runner.desc.dataset_hparams
        use_augmentation = not dataset_hparams.do_not_augment

        datasets_registry.registered_datasets[
            dataset_hparams.dataset_name].Dataset.get_train_set(use_augmentation)
        datasets_registry.registered_datasets[
            dataset_hparams.dataset_name].Dataset.get_test_set()        


    def train(self, queue=None):
        
        logging.info(f'training on client {self.client_id}')

        self.configure()
        
        lth_runner = runner_registry.get(
            self.args.subcommand).create_from_args(self.args)
        
        #run lottery
        self.platform.run_job(lth_runner.run)
        
        epoch_num = int(self.args.training_steps[0:-2])
        
        self.data_folder = os.path.join(lth_runner.desc.data_saved_folder,
                                        f'replicate_{lth_runner.replicate}')

        if "levels" in self.args:
            #lottery mode
            total_levels = self.args.levels
            target_level = total_levels
            path_to_model = os.path.join(self.data_folder,   
                        f'level_{target_level}', 'main', 
                        f'model_ep{epoch_num}_it0.pth')
        
        else:
            path_to_model = os.path.join(self.data_folder, 
                         'main', f'model_ep{epoch_num}_it0.pth')

        #init the model
        self.model = models_registry.get(
            lth_runner.desc.model_hparams, 
            outputs=lth_runner.desc.train_outputs)

        #load lottery
        self.model.load_state_dict(torch.load(path_to_model))
        weights = extract_weights(self.model)

        #set dataset number 
        self.report.set_num_samples(len(self.dataset_indices))
        self.report.weights = weights

        queue.put(self.data_folder)


    def test(self):
        pass

    def configure(self):
        """
        config: config object load from json
        """        
        self.args.client_id = self.client_id
        self.args.index_list = ' '.join(
            [str(index) for index in self.dataset_indices])

        self.platform = platforms_registry.get(
            self.args.platform).create_from_args(self.args)
