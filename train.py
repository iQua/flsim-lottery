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




def main():
    # Read configuration file
    fl_config = config.Config(args.config, args.log)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
        "lth": server.LotteryServer(fl_config) 
    }[fl_config.server]

    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Delete global model
    os.remove(fl_config.paths.model + '/global')








if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()