import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--original_json', default='/media/data/santhosh/vqa/data/data_prepro_oracle_old.json', help='path to original json file containing image lists')
parser.add_argument('--save_path', default='/media/data/santhosh/vqa/data/image_map_old_new.json', help='save path for image map')

params = vars(parser.parse_args())

lookup_dict = {}

original_json = json.load(open(params['original_json']))

for i, img in enumerate(original_json['unique_img_train']):
    lookup_dict[img] = {'idx': i + 1, 'set': 'train'}

for i, img in enumerate(original_json['unique_img_val']):
    if img not in lookup_dict:
        lookup_dict[img] = {'idx': i + 1, 'set': 'val'}

for i, img in enumerate(original_json['unique_img_test']):
    if img not in lookup_dict:
        lookup_dict[img] = {'idx': i + 1, 'set': 'test'}


json.dump(lookup_dict, open(params['save_path'], 'w'))
