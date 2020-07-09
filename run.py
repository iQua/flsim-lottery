import argparse
import client
import config
import logging
import os
import server
import sys
from datetime import datetime

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()




def main():
    """Run a federated learning simulation."""

    #set up log file
    LOG_FILE = datetime.now().strftime('./logs/log_%H_%M_%S_%d_%m_%Y.log')

    # Set logging
    logging.basicConfig(filename=LOG_FILE, 
                        format='[%(levelname)s][%(asctime)s]: %(message)s', 
                        level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')

    # Read configuration file
    fl_config = config.Config(args.config)

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
    main()
