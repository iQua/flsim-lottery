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





def episode(fl_server):

    # Run federated learning
    fl_server.run()




def main():

    agent = Agent()
    env = Environment()
    # Read configuration file
    fl_config = config.Config(args.config, args.log)

    # Initialize server
    rl_server = server.RLLotteryServer(fl_config, env, agent) 


    rl_server.boot()

    # probe clients
    rl_server.probe()


    for episode in range(1, MAX_EPISODES+1):

        episode(rl_server)
 
    








if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()