# Uses POS tagger to get nouns, and computes the statistics 
#of the types of questions present for each noun/object. 

import json
import re
import nltk
import string
import pdb
import os.path
from nltk.tokenize import word_tokenize
import string
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--question_types', default='/media/data/santhosh/VisualQuestionAnswering/VQA/QuestionTypes/mscoco_question_types.txt', help='Path to file containing question types in VQA dataset')
parser.add_argument('--raw_base_path', default='/media/data/santhosh/VisualQuestionAnswering/VQA_LSTM_CNN/data/', help='Path containing vqa raw train and val files')
parser.add_argument('--vqa_annotations_path', default='/media/data/santhosh/VisualQuestionAnswering/VQA/Annotations/', help='Path to VQA annotations')

params = vars(parser.parse_args())
# Define a function to identify the type of question
questionTypes = []

with open(params['question_types']) as inFile:
    for line in inFile: 
        questionTypes.append(string.replace(line, '\n', '').split())

numQTypes = len(questionTypes)
# Since questions can be specific like "Is this a" and less specific like "Is this"
# we first sort by descending order of lenghts of the question types and determine 
# the type in that order

questionTypes = sorted(questionTypes, key = lambda x: len(x), reverse=True)
questionTypesString = [' '.join(x) for x in questionTypes]
if not os.path.isdir('Statistics/'):
    os.mkdir('Statistics')

json.dump(questionTypesString, open('Statistics/questionTypes.json', 'w'))

print('----------------------------------\nQuestion types\n----------------------------------')

for itera in range(len(questionTypes)):
    print('%3d:'%(itera) + ' '.join(questionTypes[itera]))

print('\n')

def getQuestionType(question, qTypes):
    
    typeQ = -1
    for qNo in range(len(qTypes)): 
        q = qTypes[qNo]
        
        check = 1
        for i in range(min(len(q), len(question))):
            
            if q[i] != question[i]:
                check = 0

        if check == 1:
            typeQ = qNo
            break

    return typeQ

# Prints the statistics of the dataset
# The statistics is a dictionary, keys are nouns and 
# the value is a histogram list of question types and counts

def printStatistics(statistics):
    
    logFile = "Statistics/statistics.log"
    logger = open(logFile, "w")

    logger.write('%-15s:'%('Noun') + ' '.join(['%4d'%(qstr) for qstr in range(numQTypes)]) + '\n\n')
    for noun in statistics:
        logger.write(('%-15s:'%(noun.replace(u'\u2019', '')) + ' '.join(['%4d'%(qVal) for qVal in statistics[noun]]) + '\n'))

    logger.close()

def filterStatistics(statistics, minThresh):

    filteredStatistics = {}
    for noun in statistics:
        if sum(statistics[noun]) >= minThresh:
            filteredStatistics[noun] = statistics[noun]

    return filteredStatistics

if os.path.isfile('Statistics/statsDict.json'):
    statsDict = json.load(open("Statistics/statsDict.json"))
else:
    # Load all the data from vqa dataset: train, val and test
    base_path = params['raw_base_path']
    base_anno_path = params['vqa_annotations_path'] 
    train_json = json.load(open(base_path + 'vqa_raw_train.json', 'r'))
    val_json = json.load(open(base_path + 'vqa_raw_test.json', 'r'))
    train_anno_json = json.load(open(base_anno_path + 'mscoco_train2014_annotations.json'))
    val_anno_json = json.load(open(base_anno_path + 'mscoco_val2014_annotations.json'))

    dataList = train_json 
    annoList = train_anno_json["annotations"]
    statsDict = {}
    quesDict = {}

    # Run through the dataset, and obtain the statistics of the
    # types of questions and the numbers for each noun
    
    for elCount in range(len(dataList)):
        
        el = dataList[elCount]
        elAnno = annoList[elCount]['answers']
        # Tokenize the question and answer
        #answerSet = set()
        question = word_tokenize(el["question"].lower().replace('/', ' '))
        #for answerEl in elAnno:
        #    answer = word_tokenize(answerEl["answer"].lower().replace('/', ' '))
        #    for a in answer:
        #        answerSet.add(a)

        #answerSet = list(answerSet)
        # Run POS tagger to obtain the tags for each word
        qTagged = nltk.pos_tag(question)
        #aTagged = nltk.pos_tag(answerSet)

        # Obtain the question type for this question
        qType = getQuestionType(question, questionTypes)

        #for t in qTagged+aTagged:
        for t in qTagged:
            if t[1] == 'NN':
                if t[0] not in statsDict:
                    statsDict[t[0]] = [0 for iterX in range(numQTypes)]
                statsDict[t[0]][qType] += 1
                if t[0] not in quesDict:
                    quesDict[t[0]] = [[] for iterX in range(numQTypes)]
                quesDict[t[0]][qType].append(el['ques_id'])
    
    dataList = val_json #+ test_json[1:10]+ val_json[1:10]
    annoList = val_anno_json["annotations"]

    elCount2 = 0
    for elCount in range(len(dataList)):
        
        el = dataList[elCount]
        
        while annoList[elCount2]["question_id"] != el["ques_id"]:
            elCount2 += 1
            print(annoList[elCount2]["question_id"])

        elAnno = annoList[elCount2]['answers']
        # Tokenize the question and answer
        #answerSet = set()
        question = word_tokenize(el["question"].lower().replace('/', ' '))
        #for answerEl in elAnno:
        #    answer = word_tokenize(answerEl["answer"].lower().replace('/', ' '))
        #    for a in answer:
        #        answerSet.add(a)

        #answerSet = list(answerSet)
        # Run POS tagger to obtain the tags for each word
        qTagged = nltk.pos_tag(question)
        #aTagged = nltk.pos_tag(answerSet)

        # Obtain the question type for this question
        qType = getQuestionType(question, questionTypes)

        #for t in qTagged+aTagged:
        for t in qTagged:
            if t[1] == 'NN':
                if t[0] not in statsDict:
                    statsDict[t[0]] = [0 for iterX in range(numQTypes)]
                statsDict[t[0]][qType] += 1
                if t[0] not in quesDict:
                    quesDict[t[0]] = [[] for iterX in range(numQTypes)]
                quesDict[t[0]][qType].append(el['ques_id'])

        elCount2 += 1
    json.dump(statsDict, open('Statistics/statsDict.json', 'w'))
    json.dump(quesDict, open('Statistics/quesStatsDict.json', 'w'))

# Print out the statistics if not saved already
if not os.path.isfile('Statistics/statistics.log'):
    printStatistics(statsDict)

# If the filtered statistics are not saved already
if not os.path.isfile('Statistics/filtStatsDict.json'):
    filtStatsDict = filterStatistics(statsDict, 10)
    json.dump(filtStatsDict, open('Statistics/filtStatsDict.json', 'w'))
    printStatistics(filtStatsDict)
else:
    filtStatsDict = json.load(open('Statistics/filtStatsDict.json'))

# The features are the normalized histograms
if not os.path.isfile('Statistics/featureVectors.json'):
    featuresDict = {}
    for noun in filtStatsDict:
        norm2 = sum([pow(float(count), 2) for count in filtStatsDict[noun]]) 
        featuresDict[noun] = [float(count) / norm2 for count in filtStatsDict[noun]]
        json.dump(featuresDict, open('Statistics/featureVectors.json', 'w'))
else:
    featuresDict = json.load(open('Statistics/featureVectors.json'))
