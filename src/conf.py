import sys
import importlib
from types import SimpleNamespace
import argparse

sys.path.append("../configs")

parser = argparse.ArgumentParser(description='')

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

print("Using config file", parser_args.config)

args = importlib.import_module(parser_args.config).args

args["experiment_name"] = parser_args.config

args =  SimpleNamespace(**args)

args.img_path_train = args.data_path + 'train/'
args.img_path_val = args.data_path_2019 + 'test/'
args.img_path_test = args.data_path + 'test/'

try:
   if args.data_path_2 is not None:
      args.img_path_train_2 = args.data_path_2 + 'train/' 
except:
   args.data_path_2 = None