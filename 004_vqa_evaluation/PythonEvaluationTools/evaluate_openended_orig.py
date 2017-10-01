# coding: utf-8

import sys
dataDir = '/home/santhosh/Projects/VQA/novel-vqa-master/004_vqa_evaluation'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
#import matplotlib.pyplot as plt
#import skimage.io as io
import json
import random
import os
import pdb

novel = ''
# set up file names and paths
taskType    ='OpenEnded'
dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType ='val2014'
annFile     ='%s/Annotations/%s_%s%s_annotations.json'%(dataDir, dataType, dataSubType, novel)
quesFile    ='%s/Questions/%s_%s_%s%s_questions.json'%(dataDir, taskType, dataType, dataSubType, novel)
imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
resultType  ='lstm' #'fake'
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

# An example result json file has been provided in './Results' folder.  
[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s_%s_%s_%s%s_%s.json'%(dataDir, taskType, dataType, dataSubType, \
resultType, novel, fileType) for fileType in fileTypes]  

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

# Load question IDs of only novel word questions
quesIds = json.load(open('ques_id_hist.json'))
# Full Evaluation
vqaEval.evaluate() 
answers = []
answers.append(vqaEval.accuracy['overall']) # overall score
answers.append(vqaEval.accuracy['perAnswerType']['other']) # other type score
answers.append(vqaEval.accuracy['perAnswerType']['number']) # number type score
answers.append(vqaEval.accuracy['perAnswerType']['yes/no']) # yes/no type score

print('Ov: %.2f Oth: %.2f Num: %.2f Y/N: %.2f'%(answers[0], answers[1], answers[2], answers[3]))
