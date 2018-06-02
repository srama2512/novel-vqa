"""
Part 1:
Code to tokenize the questions and answers, tag them and 
create a count over the singular nouns present in the train dataset.
"""

import os
import sys
import pdb
import json
import nltk
import string
import argparse
import subprocess as sp
from joblib import load, dump
from random import shuffle, seed
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from nltk.tag.perceptron import PerceptronTagger

def prepro_sentence(sent):
    sent_prepro = sent.encode('utf-8').lower().translate(None, string.punctuation).strip().split()
    return sent_prepro

def filter_question(imgs, anns_train, atoi):
    new_imgs = []
    new_anns = []
    for i, img in enumerate(imgs):
        if atoi.get(img['ans'],len(atoi)+1) != len(atoi)+1:
            new_imgs.append(img)
            new_anns.append(anns_train[i])

    print 'question number reduce from %d to %d '%(len(imgs), len(new_imgs))
    return new_imgs, new_anns

def get_top_answers(imgs, params):
    if params['extern_ans_vocab'] == '':
        counts = {}
        for img in imgs:
            ans = img['ans'] 
            counts[ans] = counts.get(ans, 0) + 1

        cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
        print 'top answer and their counts:'    
        print '\n'.join(map(str,cw[:20]))
        
        vocab = []
        for i in range(params['num_ans']):
            vocab.append(cw[i][1])

        return vocab[:params['num_ans']]
    else:
        vocab = json.load(open(params['extern_ans_vocab']))
        return vocab

def prepro_question(imgs, anns, params):
  
   # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['question']
        txt = word_tokenize(s.lower().replace('/',' '))
        answerList = []
        
        for ans in anns[i]['answers']:
            ans_ws = word_tokenize(ans['answer'].lower().replace('/', ' '))
            add_check = 0
            for aS in answerList:
                if ans_ws == aS:
                    add_check = 1
                    break
            if add_check == 0:
                answerList.append(ans_ws)

        img['processed_tokens'] = txt
        # List of tokenized answers
        img['processed_ans_tokens'] = answerList
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

def union_dicts(a, b):

    for k in b:
        if k in a:
            a[k] += b[k]
        else:
            a[k] = b[k]

def get_nouns_vqa(imgs, taggers, threadidx):

    nouns_count = {}
    totalLen = len(imgs)
    count_temp = 0
    for img in imgs:
        count_temp += 1
        qTagged = taggers.tag(img['processed_tokens'])

        for w in qTagged:
            if w[1] == 'NN':
                nouns_count[w[0]] = nouns_count.get(w[0], 0) + 1
        
        for aT in img['processed_ans_tokens']:
            aTagged = taggers.tag(aT)
            for w in aTagged:
                if w[1] == 'NN':
                    nouns_count[w[0]] = nouns_count.get(w[0], 0) + 1

        if count_temp % 1000 == 0:
            print('threadIdx:%d   Progress:(%d/%d)'%(threadidx, count_temp, totalLen))

    return nouns_count

def get_nouns_vqa_parallel(imgs, numThreads=12):
    
    totalSize = len(imgs)
    partSize = int(totalSize/numThreads)
    imgsParts = []
    taggers = [PerceptronTagger() for i in range(numThreads)]
    for p in range(numThreads):
        if p < numThreads-1:
            imgsParts.append(imgs[(p*partSize):((p+1)*partSize)])
        else:
            imgsParts.append(imgs[(p+1)*partSize:totalSize])
    
    nounsPacked = Parallel(n_jobs=numThreads)(delayed(get_nouns_vqa)(imgsParts[i], taggers[i], i) for i in range(len(imgsParts)))
    
    nouns_count = {}
    nouns = []
    for p in range(numThreads):
        union_dicts(nouns_count, nounsPacked[p])
    nouns = nouns_count.keys()
    return nouns, nouns_count

def main_vqa(params):

    total_set = {}
    imgs_train = []
    imgs_test = []

    if not os.path.isfile(os.path.join(params['save_path'], 'VQA.json')):
        sp.call('mkdir %s'%(params['save_path']), shell=True)
        imgs_train = json.load(open(params['input_train_json'], 'r'))
        imgs_test = json.load(open(params['input_test_json'], 'r'))
        anns_train = json.load(open(params['input_train_annotations'], 'r'))['annotations']
        anns_test = json.load(open(params['input_test_annotations'], 'r'))['annotations']

        seed(123) # make reproducible
        # get top answers
        top_ans = get_top_answers(imgs_train, params)

        atoi = {w:i+1 for i,w in enumerate(top_ans)}
        itoa = {i+1:w for i,w in enumerate(top_ans)}

        # filter question, which isn't in the top answers.
        imgs_train, anns_train = filter_question(imgs_train, anns_train, atoi)
        zipped = zip(imgs_train, anns_train)
        shuffle(zipped)
        imgs_train, anns_train = zip(*zipped) # shuffle the order

        # tokenization and preprocessing training question
        imgs_train = prepro_question(imgs_train, anns_train, params)
        # tokenization and preprocessing testing question
        imgs_test = prepro_question(imgs_test, anns_test, params)
        VQAdata = {'imgs_train': imgs_train, 'imgs_test': imgs_test}
        json.dump(VQAdata, open(os.path.join(params['save_path'], 'VQA.json'), 'w'))
    
    else:
        print('The preprocessed data for VQA already exists @ %s'%(os.path.join(params['save_path'], 'VQA.json')))
        VQAdata = json.load(open(os.path.join(params['save_path'], 'VQA.json')))
        imgs_train = VQAdata["imgs_train"]
        imgs_test = VQAdata["imgs_test"]

    if not os.path.isfile(os.path.join(params['save_path'], "nouns_vqa.json")):
        # get the nouns from the train set
        nouns_train, nouns_train_count = get_nouns_vqa_parallel(imgs_train, 18)
        # get the nouns from the test set
        nouns_test, nouns_test_count = get_nouns_vqa_parallel(imgs_test, 18)

        total_set = {"nouns_train": nouns_train, "nouns_train_count": nouns_train_count, "nouns_test": nouns_test,  "nouns_test_count": nouns_test_count}

        json.dump(total_set, open(os.path.join(params['save_path'], "nouns_vqa.json"), 'w'))

    else:
        print('The preprocessed nouns for VQA already exists @ preprocessed/nouns_vqa.json')
        total_set = json.load(open(os.path.join(params['save_path'], "nouns_vqa.json")))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_train_json', required=True, help='input json VQA file to process into hdf5')
    parser.add_argument('--input_test_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--input_train_annotations', required=True, help='input annotation train file for novel split')
    parser.add_argument('--input_test_annotations', required=True, help='input annotation test file for novel split')
    parser.add_argument('--token_method', default='nltk', help='token method')
    parser.add_argument('--max_length', default=16, type=int, help='max length of a sentence in number of words')
    parser.add_argument('--num_ans', default=1000, type=int, help="number of answers in vqa")
    parser.add_argument('--extern_ans_vocab', default='', type=str, help='Give previously computed answer vocab')
    parser.add_argument('--save_path', default='preprocessed/', type=str, help='Path to save the preprocessed data')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)

    main_vqa(params)
