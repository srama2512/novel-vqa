import json
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--raw_train_path', required=True, type=str, help='Path to raw training json file')
parser.add_argument('--raw_test_path', required=True, type=str, help='Path to raw test json file')

opts = parser.parse_args()

raw_train = json.load(open(opts.raw_train_path))
raw_test = json.load(open(opts.raw_test_path))

print('Number of training questions: %d'%(len(raw_train)))
print('Number of testing questions: %d'%(len(raw_test)))

