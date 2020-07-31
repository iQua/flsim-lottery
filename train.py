import argparse
import logging
import os
import sys

import client
import config
import server
from rl.env import Environment
from rl.agent import MLP, ActorCritic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

def calculate_advantages(returns, values, normalize = True):
    advantages = returns - values
    if normalize:        
        advantages = (advantages - advantages.mean()) / advantages.std()        
    return advantages


def update_policy(
    policy, states, actions, log_prob_actions, \
        advantages, returns, optimizer, ppo_steps, ppo_clip):
    
    total_policy_loss = 0 
    total_value_loss = 0
    
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(
            policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
        
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
    
        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def fl_train(policy):

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

        prune_level_pred, value_pred = policy(state)

        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()


        # one round FL training
        accuracy = rl_server.round(
            round_id, action, sample_clients, level_accuracy_per_client)

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
    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_EVERY = 10
    PPO_STEPS = 5
    PPO_CLIP = 0.2


    INPUT_DIM = 30
    HIDDEN_DIM = 128
    OUTPUT_DIM = 3

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)

    LEARNING_RATE = 0.01
    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)
    
    policy.apply(init_weights)

    train_rewards = []
    test_rewards = []

    for episode in range(1, MAX_EPISODES+1):
        
        policy_loss, value_loss, train_reward = fl_train(
            policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)
        
        test_reward = evaluate(test_env, policy)
        
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        
        if episode % PRINT_EVERY == 0:
        
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
        
        if mean_test_rewards >= REWARD_THRESHOLD:
            
            print(f'Reached reward threshold in {episode} episodes')
            
            break
 

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()