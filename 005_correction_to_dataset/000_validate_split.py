from pattern.en import pluralize
import json
import pdb
from nltk.tokenize import word_tokenize
import progressbar
import json
import os
import argparse
import nltk
from nltk.tag.perceptron import PerceptronTagger

parser = argparse.ArgumentParser()
ptagger = PerceptronTagger()

parser.add_argument('--novel_words', default='../vocabs/list_of_novel_words.json', help='List of novel words in a json file')
parser.add_argument('--train_raw_novel', default='/media/data/santhosh/vqa/data/vqa_raw_train_novel_old.json', help='raw train novel json file')
parser.add_argument('--train_novel_annotations', default='/media/data/santhosh/vqa/Annotations/mscoco_train2014_novel_old_annotations.json', help='train annotations to old novel split')

params = vars(parser.parse_args())

novel = json.load(open(params['novel_words']))
novel_plu = [pluralize(p) for p in novel]

questions = json.load(open(params['train_raw_novel']))
anns = json.load(open(params['train_novel_annotations']))['annotations']
answers = [[l["answer"] for l in x["answers"]] for x in anns]

# for a, b in zip(novel, novel_plu): print a, b

num_aff = 0
num_aff_noplu = 0
num_total = 0
num_aff_ans_noplu = 0

toks = []
texts = []
if not os.path.exists('tokenizations.json'):
    bar = progressbar.ProgressBar()
    for q, ans in bar(zip(questions, answers)):
	q = q['question'].lower().replace('/', ' ')
	q_tok = word_tokenize(q)
	ans_l = []
	ans_text = []
	ans_nouns = set()
	for a in ans:
	    a = a.lower().replace('/', ' ')
	    a_tok = word_tokenize(a)
	    a_n = ptagger.tag(a_tok)
	    for cand in a_n:
	        if cand[1] == 'NN':
	            ans_nouns.add(cand[0])

	    ans_l.append(a_tok)
	    ans_text.append(a)
	toks.append((q_tok, ans_l, list(ans_nouns)))
	texts.append((q, ans_text))
    with open('tokenizations.json', 'w') as outfile:
        json.dump({'toks' : toks, 'texts' : texts}, outfile)

tokenizations = json.load(open('tokenizations.json', 'r'))
toks = tokenizations['toks']
texts = tokenizations['texts']

num_sent_total = 0
num_sent_plu = 0
num_sent_noplu = 0
num_sent_ans_noplu = 0

all_plu = []
all_noplu = []
all_ans_noplu = []
f = open('log2.txt', 'w')
bar = progressbar.ProgressBar()
for n_plu, n in bar(zip(novel_plu, novel)):
	if n == 'no': continue
	if n_plu in ['ps', 'ks', 'mrs',  'someones', 'gs', 'ms', 'his']: continue
	cnt_plu = 0
	cnt_noplu = 0
	cnt_ans_noplu = 0
	cons_plu = []
	cons_noplu = []
	cons_ans_noplu = []
	for tok, text in zip(toks, texts):
		hit_plu = -1
		hit_noplu = -1
		hit_ans_noplu = -1
		if n_plu in tok[0]:
			hit_plu = 1

		if n in tok[0]:
			hit_noplu = 1

		for a_tok in tok[1]:
			if n in a_tok:
				hit_noplu = 1
			if n_plu in a_tok:
				hit_plu = 1
		if n in tok[2]:
                        hit_ans_noplu = 1

		if hit_noplu == 1:
			if not (n, text) in cons_noplu:
				cons_noplu.append((n, text))
				cnt_noplu += 1
		if hit_plu == 1:
			if not (n_plu, text) in cons_plu:
				cons_plu.append((n_plu, text))
				cnt_plu += 1
        
                if hit_ans_noplu == 1:
                        if not (n, text, tok[2]) in cons_ans_noplu:
                                cons_ans_noplu.append((n, text, tok[2]))
                                cnt_ans_noplu += 1

	final_noplu = []
	final_plu = []
	final_ans_noplu = []
	for x in cons_noplu:
		if not x in final_noplu:
			final_noplu.append(x)
	for x in cons_plu:
		if not x in final_plu:
			final_plu.append(x) 
        for x in cons_ans_noplu:
                if not x in final_ans_noplu:
                        final_ans_noplu.append(x)

	all_plu += final_plu
	all_noplu += final_noplu
        all_ans_noplu += final_ans_noplu

	num_total += 1

	num_sent_plu += cnt_plu
	num_sent_noplu += cnt_noplu
	num_sent_ans_noplu += cnt_ans_noplu

	if cnt_plu > 0: num_aff += 1
	if cnt_noplu > 0: num_aff_noplu += 1
        if cnt_ans_noplu > 0: num_aff_ans_noplu += 1

	f.write('%20s%20s%20s%20s%20s\n' % (n, n_plu, cnt_noplu, cnt_plu, cnt_ans_noplu))

with open('log_mine_json_dump.json', 'w') as outfile:
    json.dump({'plu' : all_plu, 'noplu' : all_noplu, 'ans_noplu': all_ans_noplu}, outfile)

num_sent_total = len(toks)
print 'num_total', num_total
print 'num_aff', num_aff
print 'num_aff_noplu', num_aff_noplu
print 'num_aff_ans_noplu', num_aff_ans_noplu

print 'num_sent_total', num_sent_total
print 'num_sent_plu', num_sent_plu
print 'num_sent_noplu', num_sent_noplu
print 'num_sent_ans_noplu', num_sent_ans_noplu

f.write('num_total %d\nnum_aff %d\nnum_aff_noplu %d\n num_sent_total %d\n num_sent_plu %d\nnum_sent_noplu %d\nnum_sent_ans_noplu %d\n' % (num_total, num_aff, num_aff_noplu, num_sent_total, num_sent_plu, num_sent_noplu, num_sent_ans_noplu))
f.close()
