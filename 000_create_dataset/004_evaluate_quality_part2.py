"""
Part 2:
    Code to take the train nouns and test nouns computed in the
    previous part, compare them with the list of nouns kept
    aside from the clustering for train and test splits, and 
    get a count of the novel nouns present in the train set.
    Ideally this should be zero. But some errors in tagging or
    degenerate nouns can cause a small intersection.
"""
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='preprocessed', type=str, help='Path where the preprocessed files are stored')

params = vars(parser.parse_args())

nouns_vqa = json.load(open(os.path.join(params['save_path'], 'nouns_vqa.json')))
trainNouns = set(json.load(open('trainNouns.json')))
testNouns = set(json.load(open('testNouns.json')))

allTrainNouns = nouns_vqa['nouns_train']
allTestNouns = nouns_vqa['nouns_test']

filteredTrainNouns = set()
filteredTestNouns = set()

# Verify that train and novel words have no overlap
print('# Novel nouns in train: %d'%(len(set(allTrainNouns)&testNouns)))
print('Novel nouns in train: ', set(allTrainNouns) & testNouns)

for n in allTrainNouns:
    if n in trainNouns:
        filteredTrainNouns.add(n)

for n in allTestNouns:
    if n in trainNouns or n in testNouns:
        filteredTestNouns.add(n)

# Compute overlap statistics
print('Number of train nouns: %d'%(len(set(filteredTrainNouns))))
print('Number of test nouns: %d'%(len(set(filteredTestNouns))))
print('Number of test only nouns: %d'%(len(set(filteredTestNouns)-set(filteredTrainNouns))))
print('Number of nouns in both train and test: %d'%(len(set(filteredTestNouns) & set(filteredTrainNouns))))
    
