# Script to create novel split of the VQA dataset

import json
import random
from nltk.tokenize import word_tokenize
import nltk
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--raw_base_path', default='/media/data/santhosh/VisualQuestionAnswering/VQA_LSTM_CNN/data/', help='Base path to load the original split of VQA dataset')
parser.add_argument('--save_base_path', default='data/', help='Base path to save the novel split of VQA dataset')
parser.add_argument('--vqa_annotations_path', default='/media/data/santhosh/VisualQuestionAnswering/VQA/Annotations/', help='Path to original VQA annotations')
parser.add_argument('--save_vqa_annotations_path', default='Annotations/', help='Path to save novel VQA annotations')
parser.add_argument('--vqa_questions_path', default='/media/data/santhosh/VisualQuestionAnswering/VQA/Questions/', help='Path to original VQA question files')
parser.add_argument('--save_vqa_questions_path', default='Questions/', help='Path to save novel VQA question files')
parser.add_argument('--rng_seed', default=123, type=int, help='Random number seed')

params = vars(parser.parse_args())

# Fix the random seed for reproducibility
random.seed(params['rng_seed'])
numClusters = 14
clusterNouns = json.load(open("Clusters/clusteredNouns.json"))

# Create the known train and novel test split
clusterSplit_k_n = {}
trainNouns = set()
testNouns = set()

for i in clusterNouns:
    
    random.shuffle(clusterNouns[i])
    numOld = int(0.8*len(clusterNouns[i]))
    clusterSplit_k_n[i] = {"train": clusterNouns[i][0:numOld], "test": clusterNouns[i][(numOld+1):]}

    for n in clusterNouns[i][0:numOld]:
        trainNouns.add(n)

    for n in clusterNouns[i][(numOld+1):]:
        testNouns.add(n)

# Load the datasets
base_path = params['raw_base_path'] 
save_base_path = params['save_base_path']
base_anno_path = params['vqa_annotations_path'] 
save_base_anno_path = params['save_vqa_annotations_path'] 
base_ques_path = params['vqa_questions_path'] 
save_base_ques_path = params['save_vqa_questions_path'] 

train_json = json.load(open(base_path + 'vqa_raw_train.json', 'r'))
val_json = json.load(open(base_path + 'vqa_raw_test.json', 'r'))
train_anno_json = json.load(open(base_anno_path + 'mscoco_train2014_annotations.json'))["annotations"]
val_anno_json = json.load(open(base_anno_path + 'mscoco_val2014_annotations.json'))["annotations"]
train_q_mcq = json.load(open(base_ques_path + 'MultipleChoice_mscoco_train2014_questions.json'))
train_q_oe = json.load(open(base_ques_path + 'OpenEnded_mscoco_train2014_questions.json'))
val_q_mcq = json.load(open(base_ques_path + 'MultipleChoice_mscoco_val2014_questions.json'))
val_q_oe = json.load(open(base_ques_path + 'OpenEnded_mscoco_val2014_questions.json'))

train_kn_json = []
val_kn_json = []
train_kn_anno_json = {"info": [], "data_type": "mscoco_novel", "data_subtype": "train", "annotations": []}
val_kn_anno_json = {"info": [], "data_type": "mscoco_novel", "data_subtype": "test", "annotations": []}
train_kn_q_mcq = {"info": [], "data_type": "mscoco_novel", "data_subtype": "train", "license": [], "task_type": 'Multiple-Choice', "questions": []}
val_kn_q_mcq = {"info": [], "data_type": "mscoco_novel", "data_subtype": "test", "license": [], "task_type": 'Multiple-Choice', "questions": []}
train_kn_q_oe = {"info": [], "data_type": "mscoco_novel", "data_subtype": "train", "license": [], "task_type": 'Open-Ended', "questions": []}
val_kn_q_oe = {"info": [], "data_type": "mscoco_novel", "data_subtype": "test", "license": [], "task_type": 'Open-Ended', "questions": []}

# Create final dataset of q-a-i triplets
for elCount in range(len(train_json)):

    el = train_json[elCount]
    elAnno = train_anno_json[elCount]["answers"]
    
    #Tokenize question and answer to find out the nouns in it
    question = word_tokenize(el["question"].lower().replace('/', ' '))
    answerSet = set()
    for answerEl in elAnno:
        answer = word_tokenize(answerEl["answer"].lower().replace('/', ' '))
        for a in answer:
            answerSet.add(a)

    answerSet = list(answerSet)
    
    # Run POS tagger to obtain the tags for each word and obtain the nouns
    qTagged = nltk.pos_tag(question)
    aTagged = nltk.pos_tag(answerSet)
    
    nouns = []
    for t in qTagged+aTagged:
        if t[1] == 'NN':
            nouns.append(t[0])

    isTest = 0
    for n in nouns:
        if n in testNouns:
            isTest = 1
            break

    if not isTest:
        train_kn_json.append(el)
        train_kn_anno_json["annotations"].append(train_anno_json[elCount])
        train_kn_q_mcq["questions"].append(train_q_mcq["questions"][elCount])
        train_kn_q_oe["questions"].append(train_q_oe["questions"][elCount])
    else:
        el.pop("ans", None)
        val_kn_json.append(el)
        val_kn_anno_json["annotations"].append(train_anno_json[elCount])
        val_kn_q_mcq["questions"].append(train_q_mcq["questions"][elCount])
        val_kn_q_oe["questions"].append(train_q_oe["questions"][elCount])

elCount2 = 0
for elCount in range(len(val_json)):

    el = val_json[elCount]
    while val_anno_json[elCount2]["question_id"] != el["ques_id"]:
        elCount2 += 1
        
    elAnno = val_anno_json[elCount2]["answers"]

    # Tokenize question and answer to find out the nouns in it
    question = word_tokenize(el["question"].lower().replace('/', ' '))
    answerSet = set()

    # Since a single answer is not provided for training in validation set,
    # we select the answer that occurs the most in the annotations
    answerDictCount = {}
    for answerEl in elAnno:

        if answerEl["answer"] in answerDictCount:
            answerDictCount[answerEl["answer"]] += 1
        else:
            answerDictCount[answerEl["answer"]] = 1

    maxCount = 0
    for ans in answerDictCount:
        if answerDictCount[ans] > maxCount:
            maxCount = answerDictCount[ans]
            finalAns = ans

    # Get the set of all words in the answers to check for nouns
    for answerEl in elAnno:

        answer = word_tokenize(answerEl["answer"].lower().replace('/', ' '))
        for a in answer:
            answerSet.add(a)

    answerSet = list(answerSet)
    
    # Run POS tagger to obtain the tags for each word and obtain the nouns
    qTagged = nltk.pos_tag(question)
    aTagged = nltk.pos_tag(answerSet)
    
    nouns = []
    for t in qTagged+aTagged:
        if t[1] == 'NN':
            nouns.append(t[0])

    isTest = 0
    for n in nouns:
        if n in testNouns:
            isTest = 1
            break
    
    if not isTest:
        el["ans"] = finalAns
        train_kn_json.append(el)
        train_kn_anno_json["annotations"].append(val_anno_json[elCount2])
        train_kn_q_mcq["questions"].append(val_q_mcq["questions"][elCount2])
        train_kn_q_oe["questions"].append(val_q_oe["questions"][elCount2])
    else:
        val_kn_json.append(el)
        val_kn_anno_json["annotations"].append(val_anno_json[elCount2])
        val_kn_q_mcq["questions"].append(val_q_mcq["questions"][elCount2])
        val_kn_q_oe["questions"].append(val_q_oe["questions"][elCount2])

    elCount2 += 1

print('Size of training data: %d'%(len(train_kn_json)))
print('Size of testing data: %d'%(len(val_kn_json)))

if not os.path.isdir(save_base_path):
    os.mkdir(save_base_path)
if not os.path.isdir(save_base_anno_path):
    os.mkdir(save_base_anno_path)
if not os.path.isdir(save_base_ques_path):
    os.mkdir(save_base_ques_path)

json.dump(train_kn_json, open(save_base_path + 'train_raw_novel_2.json', 'w'))
json.dump(val_kn_json, open(save_base_path + 'val_raw_novel_2.json', 'w'))
json.dump(train_kn_anno_json, open(save_base_anno_path + 'mscoco_train2014_novel_2_annotations.json', 'w'))
json.dump(val_kn_anno_json, open(save_base_anno_path + 'mscoco_val2014_novel_2_annotations.json', 'w'))
json.dump(train_kn_q_mcq, open(save_base_ques_path + 'MultipleChoice_mscoco_train2014_novel_2_questions.json', 'w'))
json.dump(train_kn_q_oe, open(save_base_ques_path + 'OpenEnded_mscoco_train2014_novel_2_questions.json', 'w'))
json.dump(val_kn_q_mcq, open(save_base_ques_path + 'MultipleChoice_mscoco_val2014_novel_2_questions.json', 'w'))
json.dump(val_kn_q_oe, open(save_base_ques_path + 'OpenEnded_mscoco_val2014_novel_2_questions.json', 'w'))
