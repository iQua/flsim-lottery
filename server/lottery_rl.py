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

from server import LotteryServer
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
        super().__init__(config)
        self.config.server = "RL Lottery"


    def probe(self):
        logging.info('Probing all clients...')

        self.set_params(0) # 0 indicates probing diretory
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
            all_client_dict[client_id].data_folder = data_folder
            all_client_dict[client_id].report.set_num_samples(num_samples)

        self.testloader = get_testloader(
            self.config.lottery_args.dataset_name, self.server_indices) 
        
        reports = self.reporting(self.clients)

        tot_level = self.config.lottery_args.levels + 1
        ep_num = int(self.config.lottery_args.training_steps[0:-2])

        level_accuracy_per_client = {}
        for client in self.clients:
            level_accuracy_per_client[client.client_id] = []
            unpruned_accuracy = 0

            for lvl in range(tot_level):
                lvl_accuracy = self.__get_lvl_accuracy(os.path.join(
                    client.data_folder, f'level_{lvl}', 'main', 'logger'))                

                if lvl == 0: unpruned_accuracy = lvl_accuracy
                
                level_accuracy_per_client[client.client_id].append(\
                    round((lvl_accuracy - unpruned_accuracy) * 10000, 2))
        
        return level_accuracy_per_client


    def __get_lvl_accuracy(self, logger_path):
        with open(logger_path, 'r') as logger_fd:
            lines = logger_fd.readlines()

        return float(lines[-3].split(',')[-1])


    def round(self, round_id, prune_level, sample_clients, \
            level_accuracy_per_client):
        
        train_mode = self.config.lottery_args.subcommand
        self.config.lottery_args.levels = prune_level # set client prune level

        self.set_params(round_id) # 0 indicates probing diretory
        
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
        
        reports = self.reporting(sample_clients)

        tot_level = prune_level + 1
        ep_num = int(self.config.lottery_args.training_steps[0:-2])

        # update level_accuracy_per_client
        for client in sample_clients:
            unpruned_accuracy = 0

            for lvl in range(tot_level):
                lvl_accuracy = self.__get_lvl_accuracy(os.path.join(
                    client.data_folder, f'level_{lvl}', 'main', 'logger'))                
                
                if lvl == 0: unpruned_accuracy = lvl_accuracy
                
                level_accuracy_per_client[client.client_id][lvl] = \
                    round((lvl_accuracy - unpruned_accuracy) * 10000, 2)

        return self.get_pruned_model(sample_clients, reports, prune_level)


    def get_pruned_model(self, sample_clients, reports, prune_level):
        # accuracy_dict = { level: global_model_accuracy }
        accuracy_dict = self.get_accuracy_per_level(sample_clients, reports)
        selected_model_path = os.path.join(self.global_model_path_per_round, \
            'global', f'level_{prune_level}', 'model.pth')
        
        self.model.load_state_dict(torch.load(selected_model_path))
        self.model.eval()

        # update static global model for next round
        self.save_model(self.model, self.static_global_model_path)
        # backup the seleted global model to round directory
        # self.save_model(self.model, self.global_model_path_per_round)

        accuracy = accuracy_dict[prune_level]
        logging.info(f'Selected level-{prune_level} model accuracy: '\
            + '{:.2f}%'.format(100 * accuracy))
        
        return accuracy
          

    def train_best_model_rl(self, sample_clients, reports):
        pass
    

    def get_best_model_rl(self, sample_clients, reports):
        pass


    def is_done(self, round_id, current_accuracy):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy

        return (current_accuracy >= target_accuracy) or (round_id >= rounds)
