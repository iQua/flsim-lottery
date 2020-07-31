import argparse
import logging
import os
import sys

import client
import config
import server
from rl.env import Environment
from rl.agent import Agent

import torch


MAX_EPISODES = 1000


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()


def calculate_reward():

    pass


def fl_train():

    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    episode_reward = 0
    done = False
    
    # Read configuration file
    fl_config = config.Config(args.config, args.log)

    rounds = fl_config.fl.rounds
    target_accuracy = fl_config.fl.target_accuracy

    # Initialize server
    rl_server = server.RLLotteryServer(fl_config) 
    # env = Environment(rl_server)

    rl_server.boot()

    # probe clients
    level_accuracy_per_client = rl_server.probe()

    logging.info(f'Probing clients: {level_accuracy_per_client}')

    for round_id in range(1, rounds+1):
        logging.info('**** Round {}/{} ****'.format(round_id, rounds))
    
        sample_clients = rl_server.selection()

        state = [level_accuracy_per_client[client.client_id] \
                for client in sample_clients]
        states.append(state)

        prune_level_pred, value_pred = make_action(state)

        # one round FL training
        accuracy = rl_server.round(
            round_id, prune_level, sample_clients, level_accuracy_per_client)

        logging.info(f'Round {round_id} accuracy: {accuracy}')
        logging.info(f'Round {round_id}: {level_accuracy_per_client}')

        done = target_accuracy and (accuracy >= target_accuracy)

        reward = accuracy - target_accuracy # negative value
        rewards.append(reward)
        episode_reward += reward        

        if done: # Break loop when target accuracy is met
            logging.info('Target accuracy reached.')
            break
    

    returns = calculate_returns(rewards, discount_factor)

    return episode_reward, 

def main():

    agent = Agent()
    
    for episode in range(0, 1):
        fl_train()
 

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()