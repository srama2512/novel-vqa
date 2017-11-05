import json

nouns_vqa = json.load(open('preprocessed/nouns_vqa.json'))
trainNouns = set(json.load(open('preprocessed/trainNouns.json')))
testNouns = set(json.load(open('preprocessed/testNouns.json')))

allTrainNouns = nouns_vqa['nouns_train']
allTestNouns = nouns_vqa['nouns_test']

filteredTrainNouns = set()
filteredTestNouns = set()

# Verify that train and novel words have no overlap
#print(set(allTrainNouns) & testNouns)
#assert(len(set(allTrainNouns) & testNouns) == 0)
#print('Train and novel nouns have 0 overlap!')
print('Novel nouns in train: %d'%(len(set(allTrainNouns)&testNouns)))
print(set(allTrainNouns) & testNouns)

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
    
