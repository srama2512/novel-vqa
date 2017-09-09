# Preprocesses the data file, where each line is a piece of text
# The preprocessed data is then saved as a hdf5 file

import string
import h5py
import pdb
import argparse
import numpy as np
import json
from random import shuffle, seed
from joblib import Parallel, delayed
import time
import os

seed(123) # To make reproducable
# Function to preprocess a sentence and output words in a list
def prepro_sentence(sents):
	
    sents_prepro = [''.join([l for l in sents[sentIdx] if ord(l) < 128]).encode('utf-8').lower().translate(None, string.punctuation).strip().split() for sentIdx in xrange(len(sents))]
    return sents_prepro

def prepro_sentence_parallel(sents, nThreads=12):
    
    sentsSplit = []
    totalSize = len(sents)
    partSize = int(len(sents)/nThreads)

    for p in range(nThreads):
        if p < nThreads-1:
            sentsSplit.append(sents[p*partSize:(p+1)*partSize])
        else:
            sentsSplit.append(sents[p*partSize:totalSize])
    
    sentsPacked = Parallel(n_jobs=nThreads)(delayed(prepro_sentence)(sentsSplit[i]) for i in range(len(sentsSplit)))
    sentsReturn = []
    
    for p in range(nThreads):
        sentsReturn += sentsPacked[p]

    return sentsReturn

def insertUNK(sents, vocab, threadIdx):
   
    sentsMod = []
    totalLen = len(sents)
    for i in xrange(totalLen):
        sentsMod.append([w if w in vocab else 'UNK' for w in sents[i]])
        if i%1000 == 0:
            print('threadIdx: %d    Progress: (%d/%d)'%(threadIdx, i, totalLen))

    return sentsMod

def insertUNKParallel(sents, vocabList, nThreads=12):
    
    vocab = set(vocabList) 
    sentsSplit = []
    totalSize = len(sents)
    partSize = int(len(sents)/nThreads)
    vocabs = [vocab for i in range(nThreads)]

    for p in range(nThreads):
        if p < nThreads-1:
            sentsSplit.append(sents[p*partSize:(p+1)*partSize])
        else:
            sentsSplit.append(sents[p*partSize:totalSize])
    
    sentsPacked = Parallel(n_jobs=nThreads)(delayed(insertUNK)(sentsSplit[i], vocabs[i], i) for i in range(len(sentsSplit)))
    sentsReturn = []
    
    for p in range(nThreads):
        sentsReturn += sentsPacked[p]

    return sentsReturn

# Function to create the vocabularyfrom the dataset
def create_vocab(dataset, params):
    
    if params['ext_vocab'] == '': 
        count_thr = params['word_count_threshold']

        # stores the counts of different words
        word_count = {}
        for sent in dataset['tokenized']:
            for word in sent:
               word_count[word] = word_count.get(word, 0) + 1
        

        word_count_sorted = sorted([(count, w) for w, count in word_count.iteritems()], reverse=True)
        
        print ''.join(["-" for itera in range(15)]) + '\n' + "Top words and their counts"
        print ''.join(["-" for itera in range(15)])
        print '\n'.join(map(str, word_count_sorted[:20]))

        total_words = sum(word_count.itervalues())
        print ''.join(["-" for itera in range(15)])
        print 'Total words:', total_words

        vocab = set()
        if not (params['vqa_vocab'] == ''):
            print('Adding words from vqa vocabulary')
            vqa_vocab = json.load(open(params['vqa_vocab']))
            vocab.update(vqa_vocab)
        
        if not (params['novel_vocab'] == ''):
            print('Adding novel words from vqa dataset')
            novel_vocab = json.load(open(params['novel_vocab']))
            vocab.update(novel_vocab)

        vocab_update = [w for w, n in word_count.iteritems() if n > count_thr]
        unk_words = [w for w, n in word_count.iteritems() if (n <= count_thr) and w not in vocab]
        
        if len(vocab_update) > params["max_vocab_size"]:
            word_count_new_sorted = sorted([(word_count[w], w) for w in vocab_update], reverse=True)
            vocab_new = [word_count_new_sorted[i][1] for i in range(params["max_vocab_size"])]
            unk_words = unk_words + [word_count_new_sorted[i][1] for i in range(params["max_vocab_size"], len(word_count_new_sorted))]
            vocab_update = vocab_new

        for w in vocab:
            if w in unk_words:
                unk_words.remove(w)
        
        vocab.update(vocab_update)
        vocab = list(vocab)

        unk_count = sum(word_count[w] for w in unk_words)
        print 'Number of bad words: %d/%d = %.2f%%' % (len(unk_words), len(word_count), len(unk_words)*100.0/len(word_count))
        print 'Number of words in vocab: %d' %(len(vocab))
        print 'Number of UNKs: %d/%d = %.2f%%' % (unk_count, total_words, unk_count*100.0/total_words)

        # Lets look at the distribution of lengths as well
        sent_lengths = {}
        for sent in dataset['tokenized']:
            nw = len(sent)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1 
        
        max_len = max(sent_lengths.keys())
        print 'max length sentence in raw data: ', max_len
        print 'sentence length distribution (count, number of words):'
        sum_len = sum(sent_lengths.values())
        for i in xrange(min(max_len+1, 100)):
            print '%2d: %10d   %.4f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len)

        # lets now produce the final annotations
        if unk_count > 0:
            # additional special UNK token we will use below to map infrequent words to
            print 'inserting the special UNK token'
            vocab.append('UNK')

        # Note: The end of generation token is simply a 0-vector, as per neuraltalk2 convention
        # The start token will be len(vocab)+1, and will be used during the forward pass by default
        
        # Used as an input to the encoder and decoder, appropriate start and end tokens will be used
        # during the forward pass. This will also be the label for the decoder. 
        dataset['final'] = [] 
        for sent in dataset['tokenized']:
            sent_final = [w if w in vocab else 'UNK' for w in sent]
            dataset['final'].append(sent_final)
    else:
    
        print('Found external vocabulary')
        genStart = time.time()
        vocab = json.load(open(params['ext_vocab']))
        print('Loaded vocabulary: %.3f', time.time()-genStart)
        print('Inserting UNKs')
        dataset['final'] = insertUNK(dataset['tokenized'], set(vocab), 1)
        print('Inserted UNKs')

    return vocab

# Function to encode the sentences as sequence of integers
def encode_sentences(dataset, params, wtoi):

    max_length = params['max_length']
    input_sentence = {'train':[], 'val':[], 'test':[]}
    # Records the length of the input to encoder, decoder
    input_length = {'train':[], 'val':[], 'test':[]}

    # Maps sentences to sequence of integers
    # Train part of dataset

    for sentIdx in range(len(dataset['final'])):
        sent = dataset['final'][sentIdx]
        Ltemp = np.zeros((1, max_length), dtype='uint32')
        lengthTemp = np.zeros((1), dtype='uint32')
        splitCurr = dataset['split'][sentIdx]
        for k, w in enumerate(sent):
            if k < max_length:
                Ltemp[0, k] = wtoi[w]
        input_sentence[splitCurr].append(Ltemp)
        lengthTemp[0] = min(max_length, len(sent)) 
        input_length[splitCurr].append(lengthTemp)

    L = {}
    L['train'] = np.concatenate(input_sentence['train'], axis=0)
    L['val'] = np.concatenate(input_sentence['val'], axis=0)
    L['test'] = np.concatenate(input_sentence['test'], axis=0)

    input_length['train'] = np.concatenate(input_length['train'], axis=0)
    input_length['val'] = np.concatenate(input_length['val'], axis=0)
    input_length['test'] = np.concatenate(input_length['test'], axis=0)

    assert np.all(input_length > 0), 'Error: Some captions had no words!'
    return L, input_length


BookCorpusDataset = {}
BookCorpusDataset['unprocessed'] = []
BookCorpusDataset['tokenized'] = []

WikiCorpusDataset = {}
WikiCorpusDataset['unprocessed'] = []
WikiCorpusDataset['tokenized'] = []

parser = argparse.ArgumentParser()

# options
parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
parser.add_argument('--max_length', default=16, type=int, help='max length of a sentence in number of words')
parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
parser.add_argument('--output_json', default='data.json', help='output json file')
parser.add_argument('--num_val', default=30000, type=int, help='number of validation sentences')
parser.add_argument('--num_test', default=100000, type=int, help='number of test sentences')
parser.add_argument('--max_vocab_size', default=20000, type=int, help='maximum size of vocabulary')
parser.add_argument('--ext_vocab', default='', help='external vocabulary to use')
parser.add_argument('--book_corpus_path1', default='/media/data/santhosh/VisualQuestionAnswering/BookCorpus/books_large_p1.txt', help='path to part 1 of book corpus')
parser.add_argument('--book_corpus_path2', default='/media/data/santhosh/VisualQuestionAnswering/BookCorpus/books_large_p2.txt', help='path to part 2 of book corpus')
parser.add_argument('--wiki_corpus_path', default='/media/data/santhosh/VisualQuestionAnswering/BookCorpus/books_large_p3.txt', help='path to wiki corpus')


args = parser.parse_args()
params = vars(args)

datasetPath1 = params['book_corpus_path1'] 
datasetPath2 = params['book_corpus_path2'] 
datasetPath3 = params['wiki_corpus_path']

# Read dataset and preprocess it
if not os.path.isfile('preprocessed/BookCorpusPlusWiki.json'):
    print('\nReading the data and preprocessing')
    startTime = time.time()

    with open(datasetPath1) as inFile:
        for line in inFile:
                    BookCorpusDataset['unprocessed'].append(str.replace(line, '\n', ''))

    with open(datasetPath2) as inFile:
        for line in inFile:
                    BookCorpusDataset['unprocessed'].append(str.replace(line, '\n', ''))

    with open(datasetPath3) as inFile:
        for line in inFile:
                    WikiCorpusDataset['unprocessed'].append(str.replace(line, '\n', ''))

    print('Finished reading dataset: %.3f secs'%(time.time()-startTime))

    # Randomized shuffling of the dataset
    print('Random shuffling of dataset started')
    startTime = time.time()
    shuffle(BookCorpusDataset['unprocessed'])
    shuffle(WikiCorpusDataset['unprocessed'])
    print('Finished shuffling: %.3f secs'%(time.time()-startTime))

    BookCorpusDataset['unprocessed'] += WikiCorpusDataset['unprocessed']

    print('Preprocessing dataset')
    startTime = time.time()
    BookCorpusDataset['tokenized'] = prepro_sentence_parallel(BookCorpusDataset['unprocessed'], 22)
    json.dump(BookCorpusDataset, open('preprocessed/BookCorpusPlusWiki.json', 'w'))

    print('Finished preprocessing: %.3f secs'%(time.time()-startTime))
else:
    print('Already preprocessed BookCorpus and Wiki')
    startTime = time.time()
    BookCorpusDataset = json.load(open('preprocessed/BookCorpusPlusWiki.json'))
    print('Loaded preprocessed dataset: %.3f secs'%(time.time()-startTime))

# Create the vocabulary of the dataset
print('Creating vocabulary and inserting UNKs')
startTime = time.time()
BookCorpusVocab = create_vocab(BookCorpusDataset, params)
itow = {i+1:w for i, w in enumerate(BookCorpusVocab)}
wtoi = {w:i+1 for i, w in enumerate(BookCorpusVocab)}
print('Finished creating vocabulary and inserting UNKs: %.3f secs'%(time.time()-startTime))

# Assign splits to the dataset
BookCorpusDataset['split'] = []
for i in range(len(BookCorpusDataset['tokenized'])):
    
    if i < params['num_val']:
        BookCorpusDataset['split'].append('val')
    elif i < params['num_val'] + params['num_test']:
        BookCorpusDataset['split'].append('test')
    else:
        BookCorpusDataset['split'].append('train')

# Encode sentences using the vocabulary
L, input_length = encode_sentences(BookCorpusDataset, params, wtoi)

# Save the dataset as a h5 file
f = h5py.File(params['output_h5'], "w")
f.create_dataset("labels/train", dtype='uint32', data=L['train'])
f.create_dataset("labels/val", dtype='uint32', data=L['val'])
f.create_dataset("labels/test", dtype='uint32', data=L['test'])
f.create_dataset("label_length/train", dtype='uint32', data=input_length['train'])
f.create_dataset("label_length/val", dtype='uint32', data=input_length['val'])
f.create_dataset("label_length/test", dtype='uint32', data=input_length['test'])

f.close()

print 'wrote ', params['output_h5']

# create output json file
out = {}
out['num_test'] = params['num_test']
out['num_val'] = params['num_val']
out['num_train'] = len(BookCorpusDataset['tokenized'])-params['num_test']-params['num_val']
out['ix_to_word'] = itow # encode the (1-indexed) vocab
json.dump(out, open(params['output_json'], 'w'))
print 'wrote ', params['output_json']
