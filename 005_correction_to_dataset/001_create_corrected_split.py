import json
from pattern.en import pluralize
import pdb
from nltk.tokenize import word_tokenize
import nltk
import progressbar
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--novel_words', default='../vocabs/list_of_novel_words.json', help='List of novel words in a json file')
parser.add_argument('--train_base_raw', default='/media/data/santhosh/vqa/data/', help='train raw novel directory path')
parser.add_argument('--base_annotations', default='/media/data/santhosh/vqa/Annotations/', help='train annotation for old novel split')
parser.add_argument('--base_questions', default='/media/data/santhosh/vqa/Questions/', help='train questions for old novel split')
parser.add_argument('--save_base_raw', default='/media/data/santhosh/vqa/data/')
parser.add_argument('--save_base_annotations', default='/media/data/santhosh/vqa/Annotations/', help='path to save train annotations for new novel split')
parser.add_argument('--save_base_questions', default='/media/data/santhosh/vqa/Questions/', help='path to save train questions for new novel split')

params = vars(parser.parse_args())

rem_words = ['p', 'mr', 'k', 'someone', 'g', 'm', 'hi', 'no']
novel_words = [x for x in json.load(open(params['novel_words'])) if not x in rem_words]
novel_words_set = set(novel_words)

# Load the datasets
train_raw_questions = json.load(open(params['train_base_raw'] + 'vqa_raw_train_novel_old.json'))
train_annotations = json.load(open(params['base_annotations'] + 'mscoco_train2014_novel_old_annotations.json'))['annotations']
train_oe_questions = json.load(open(params['base_questions'] + 'OpenEnded_mscoco_train2014_novel_old_questions.json'))
train_mcq_questions = json.load(open(params['base_questions'] + 'MultipleChoice_mscoco_train2014_novel_old_questions.json'))
test_raw_questions_path = params['train_base_raw'] + 'vqa_raw_test_novel_old.json'
test_annotations_path = params['base_annotations'] + 'mscoco_val2014_novel_old_annotations.json'
test_oe_questions_path = params['base_questions'] + 'OpenEnded_mscoco_val2014_novel_old_questions.json'
test_mcq_questions_path = params['base_questions'] + 'MultipleChoice_mscoco_val2014_novel_old_questions.json'

# Create file paths for saving dataset
save_train_raw_questions_path = params['save_base_raw'] + 'vqa_raw_train_novel_new.json' 
save_train_annotations_path = params['save_base_annotations'] + 'mscoco_train2014_novel_new_annotations.json'
save_train_oe_questions_path = params['save_base_questions'] + 'OpenEnded_mscoco_train2014_novel_new_questions.json'
save_train_mcq_questions_path = params['save_base_questions'] + 'MultipleChoice_mscoco_train2014_novel_new_questions.json'
save_test_raw_questions_path = params['save_base_raw'] + 'vqa_raw_test_novel_new.json'
save_test_annotations_path = params['save_base_annotations'] + 'mscoco_val2014_novel_new_annotations.json'
save_test_oe_questions_path = params['save_base_questions'] + 'OpenEnded_mscoco_val2014_novel_new_questions.json'
save_test_mcq_questions_path = params['save_base_questions'] + 'MultipleChoice_mscoco_val2014_novel_new_questions.json'

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

save_train_raw_questions = []
save_train_annotations = {"info": [], "data_type": "mscoco_novel", "data_subtype": "train", "annotations":[]}
save_train_oe_questions = {"info": [], "data_type": "mscoco_novel", "data_subtype": "train", "licence": [], "task_type": "Open-Ended", "questions": []}
save_train_mcq_questions = {"info": [], "data_type": "mscoco_novel", "data_subtype": "train", "licence": [], "task_type": "Multiple-Choice", "questions": []}

for elCount in bar(range(len(train_raw_questions))):
    # print('Processing question %d/%d'%(elCount, len(train_raw_questions)))
    el = train_raw_questions[elCount]
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
        save_train_raw_questions.append(el)
        save_train_annotations["annotations"].append(train_annotations[elCount])
        save_train_oe_questions["questions"].append(train_oe_questions["questions"][elCount])
        save_train_mcq_questions["questions"].append(train_mcq_questions["questions"][elCount])

    count_plural_issue += isTestPlural

# Save the new dataset
with open(save_train_raw_questions_path, 'w') as outfile:
    json.dump(save_train_raw_questions, outfile)

with open(save_train_annotations_path, 'w') as outfile:
    json.dump(save_train_annotations, outfile)

with open(save_train_oe_questions_path, 'w') as outfile:
    json.dump(save_train_oe_questions, outfile)

with open(save_train_mcq_questions_path, 'w') as outfile:
    json.dump(save_train_mcq_questions, outfile)

# Since the test set remains the same, just copy the files
# to the right path
os.system('cp %s %s'%(test_raw_questions_path, save_test_raw_questions_path))
os.system('cp %s %s'%(test_annotations_path, save_test_annotations_path))
os.system('cp %s %s'%(test_oe_questions_path, save_test_oe_questions_path))
os.system('cp %s %s'%(test_mcq_questions_path, save_test_mcq_questions_path))

print 'Number of plural train questions', count_plural_issue
