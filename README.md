# An Empirical Evaluation of Visual Question Answering for Novel Objects

This repository contains the dataset creation and model generation files for our [work](https://arxiv.org/abs/1704.02516) 
published in CVPR 2017. For more details, please refer to the [project page](http://san25dec.github.io/publications/novel-vqa-cvpr17).
This training code has been borrowed and modified from [here](https://github.com/VT-vision-lab/VQA_LSTM_CNN) and [here](https://github.com/karpathy/neuraltalk2). 
The VQA evaluation scripts have been borrowed from [here](https://github.com/VT-vision-lab/VQA/). 

# Requirements

1. [Torch7](https://github.com/torch/torch7)
2. [NLTK](http://www.nltk.org/) in Python2.7

# Installation

This code has been written for Ubuntu. Some of the necessary packages (borrowed from the two codebases mentioned earlier). 
```
sudo apt-get install libpng-dev libtiffio-dev libhdf5-dev
pip install pillow
pip install -r requirements.txt
python -c "import nltk; nltk.download('all')"
```

# Preparing the dataset
The train and validation splits of VQA dataset have been modified to give novel train and validation splits as explained in our
paper. The dataset provided here is a slightly modified version of the originally proposed dataset. A small percentage (5%) of
questions have been removed from the training set as explained in our [project page](http://san25dec.github.io/publications/novel-vqa-cvpr17).

1. Download the real images section of the [VQA v1 dataset](http://www.visualqa.org/vqa_v1_download.html). 
2. Download the train2014 and val2014 images and annotations from [MS-COCO](http://cocodataset.org/#download) and store them as
instructed in https://github.com/VT-vision-lab/VQA/ .
3. Clone this code and go to `000_create_dataset`. 
```
git clone https://github.com/san25dec/novel-vqa.git
cd novel-vqa/000_create_dataset
cp 000_vqa_preprocessing.py <VQA_DATA_DIR>
cd <VQA_DATA_DIR>
python vqa_preprocessing.py --download False --split 1
```
This code outputs `vqa_raw_train.json` and `vqa_raw_test.json`. 

4. Run the remaining scripts in `000_create_dataset` in the sequential order specified by setting the correct path parameters.

Note that the random seed was initially not set for generating the data. As a result, the exact data may not be replicable.
To ensure reproducability, we have provided the data split that we obtained [here](https://drive.google.com/drive/folders/0B_zNnwTk0MVydVRtQTgtbTl5M0k?usp=sharing).
The files without any special tagging are the original VQA data split. The files with `_novel_old` tag are the splits used for
our CVPR acceptance paper. The files with `_novel_new` tag are the corrected splits with 5% of questions removed. The code
to correct the dataset is provided in `005_correction_to_dataset`. 

# Generated Vocabularies 
We have provided the vocabularies we used for all our experiments in `vocabs/`. It contains the train (`vocab_train.json`), 
oracle (`vocab_oracle.json`) and general (`vocab_general.json`) vocabularies. It also contains the list of novel words which
we used (`list_of_novel_words.json`).


