# Script to cluster nouns into similar groups based on question statistics.
# Since random seed was not added, the experiment may not be repeatable.
# The cluster assignments used in the paper has also been added to the
# repository for repeatability.

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import k_means_
import matplotlib.pyplot as plt

import json
import pdb
import numpy as np
from time import time
import os.path

# Defining chi squared distance for comparing histograms

features = json.load(open('Statistics/featureVectors.json'))
kVal = []
inertiaVal = []
'''
for numC in [i for i in range(2, 50)]:
    estimator = KMeans(init='k-means++', n_clusters=numC, n_init=400, max_iter=5000, n_jobs=12)
    data = {}

    data["features"] = np.empty((len(features), len(features[features.keys()[0]])))
    data["names"] = []

    count = 0
    for f in features:

        data["features"][count] = np.array(features[f])
        data["names"].append(f)
        count += 1

    t0 = time()
    estimator.fit(data["features"])
    
    # Inertia is the minimization criterion
    # Silhouette score is a measure of how good the clusters are, higher the better
    kVal.append(numC)
    inertiaVal.append(estimator.inertia_)
    print('% 9s    numC:%d     inertia: %i    silScore: %.3f' % ('KMeans', numC, estimator.inertia_, metrics.silhouette_score(data["features"], estimator.labels_, metric='euclidean', sample_size=len(features))))

plt.clf()
plt.plot(kVal, inertiaVal, label = 'Inertia (vs) k')
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()

pdb.set_trace()
''' and None

#Best numC was selected as 6 based on the inertia and silScore
if not os.path.isdir('Clusters/'):
    os.mkdir('Clusters')

if not os.path.isfile('Clusters/clusteredNouns.json'):
    numC = 14
    estimator = KMeans(init='k-means++', n_clusters=numC, n_init=400, max_iter=5000, n_jobs=12)
    data = {}

    data["features"] = np.empty((len(features), len(features[features.keys()[0]])))
    data["names"] = []

    count = 0
    for f in features:

        data["features"][count] = np.array(features[f])
        data["names"].append(f)
        count += 1

    t0 = time()
    estimator.fit(data["features"])

    # Inertia is the minimization criterion
    # Silhouette score is a measure of how good the clusters are, higher the better
    print('% 9s    numC:%d     inertia: %i    silScore: %.3f' % ('KMeans', numC, estimator.inertia_, metrics.silhouette_score(data["features"], estimator.labels_, metric='euclidean', sample_size=len(features))))

    clusteredNouns = {}
    clusterCenters = {}

    for labelNo in range(len(estimator.labels_)):

        label = estimator.labels_[labelNo]
        
        if str(label) not in clusteredNouns:
            clusteredNouns[str(label)] = []
        
        clusteredNouns[str(label)].append(data["names"][labelNo])

    for c in range(len(estimator.cluster_centers_)):

        clusterCenters[str(c)] = estimator.cluster_centers_[c].tolist()

    json.dump(clusteredNouns, open('Clusters/clusteredNouns.json', 'w'))
    json.dump(clusterCenters, open('Clusters/clusterCenters.json', 'w'))

else:
    
    clusteredNouns = json.load(open('Clusters/clusteredNouns.json'))
    clusterCenters = json.load(open('Clusters/clusterCenters.json'))

clusterLogger = open('Clusters/ClusterStatistics.txt', 'w')
questionTypes = json.load(open('Statistics/questionTypes.json'))
filtStatsDict = json.load(open('Statistics/filtStatsDict.json'))

for i in range(len(clusterCenters)):
    
    clusterLogger.write(''.join(['-' for j in range(10)]) + '\nCluster %d\n'%(i) + ''.join(['-' for j in range(10)]) + '\n')
    top5Q = sorted(range(len(clusterCenters[str(i)])), key=lambda it: clusterCenters[str(i)][it], reverse=True)[:5]
    top5Scores = sorted(clusterCenters[str(i)], reverse=True)[:5]
    clusterLogger.write('Top 5 question types: ' + '; '.join([questionTypes[j] for j in top5Q]) + '\n')
    clusterLogger.write('Top 5 cluster scores: ' + '; '.join(['%.3f'%(j) for j in top5Scores]) + '\n')
    clusterLogger.write(''.join(['-' for j in range(5)]) + 'Nouns associated' + ''.join(['-' for j in range(5)]) + '\n')
    for noun in clusteredNouns[str(i)]:
        top5QNoun = sorted(range(len(filtStatsDict[noun])), key=lambda it: filtStatsDict[noun][it], reverse=True)[:5]
        clusterLogger.write('%-15s'%(noun.replace(u'\u2019', '')) + ': ' + '; '.join([questionTypes[j] for j in top5QNoun]) + '\n')

    clusterLogger.write('\n')

