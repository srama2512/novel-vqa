import json
from pattern.en import pluralize
import pdb
from nltk.tokenize import word_tokenize
import nltk
import progressbar
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--novel_words', default='../vocabs/list_of_novel_words.json', help='List of novel words in a json file')
parser.add_argument('--train_questions', default='/media/data/santhosh/vqa/data/vqa_raw_train_novel_old.json', help='train raw novel json file')
parser.add_argument('--train_annotations', default='/media/data/santhosh/vqa/Annotations/mscoco_train2014_novel_old_annotations.json', help='train annotation for old novel split')
parser.add_argument('--save_train_split', default='/media/data/santhosh/vqa/data/vqa_raw_train_novel_new.json')

params = vars(parser.parse_args())

rem_words = ['p', 'mr', 'k', 'someone', 'g', 'm', 'hi', 'no']
novel_words = [x for x in json.load(open(params['novel_words'])) if not x in rem_words]
novel_words_set = set(novel_words)
train_questions = json.load(open(params['train_questions']))
train_annotations = json.load(open(params['train_annotations']))['annotations']

pluralized_to_orig = {}
pluralized_noun_words = set()

bar = progressbar.ProgressBar()

for i, word in bar(enumerate(novel_words)):
    pluralized_word = pluralize(word)
    if pluralized_word != word:
        pluralized_to_orig[pluralized_word] = word
        pluralized_noun_words.add(pluralized_word)
    else:
        print('Plural = word for %s'%(word))

count_plural_issue = 0

set_plu = {}
bar = progressbar.ProgressBar()

train_instances = []

for elCount in bar(range(len(train_questions))):
    # print('Processing question %d/%d'%(elCount, len(train_questions)))
    el = train_questions[elCount]
    elAnno = train_annotations[elCount]["answers"]
    text_q = el["question"].lower().replace('/', ' ')
    question_tokenized = word_tokenize(text_q)
    answerSet = set()
    text_ans = []
    for answerEl in elAnno:
        a = answerEl["answer"].lower().replace('/', ' ')
        text_ans.append(a)
        answer = word_tokenize(a)
        for a in answer:
            answerSet.add(a)

    text = (text_q, text_ans)
    answerList = list(answerSet)
    
    isTestPlural = 0

    for qWord in question_tokenized + answerList:
        if qWord in pluralized_noun_words:
            isTestPlural = 1
            break 

    if isTestPlural != 1:
        train_instances.append(el)

    count_plural_issue += isTestPlural

with open(params['save_train_split'], 'w') as outfile:
    json.dump(train_instances, outfile)

print 'Number of plural train questions', count_plural_issue
