#coding:utf-8
from __future__ import print_function, division
import os
import re
import numpy as np
import random
from constants import *
from word_embedding import load_word2vec, embedding, embedding_whole
try:
    import xml.etree.cElementTree as ET 
except ImportError:
    import xml.etree.ElementTree as ET

def preprocess_string(s, tag):
    if s[-1] == '\n':
        s = s[:-1]
    s = s.replace('\t', '')
    if "<review>" not in s and "<reviews>" not in s and "</review>" not in s and "</reviews>" not in s:
        s = s.replace('<', '&lt;')
        s = s.replace('>', '&gt;')
    if tag == CN:
        s.replace('<<', '《')
        s.replace('>>', '》')
    return re.sub(r'&(?!(lt;)|(gt;)|(amp;)|(apos;)|(quot;))', '&amp;', s)
    
def preprocess_file(fname, tag):
    with open(fname, 'r') as f_in:
        lines = f_in.readlines()
        lines = list(filter(lambda x:x != '\n', lines))
        for i in range(len(lines)):
            lines[i] = preprocess_string(lines[i], tag)
            
    with open(fname, 'w') as f_out:
        for line in lines:
            f_out.write(line + '\n')

def cvt_to_npz(tag):
    model = load_word2vec(tag)
    xmltree_n = ET.parse(os.path.join(Dataset_Dir, Tag_Name[tag], "%s_negative.xml" % Tag_Name[tag]))
    xmlroot_n = xmltree_n.getroot()
    xmltree_p = ET.parse(os.path.join(Dataset_Dir, Tag_Name[tag], "%s_positive.xml" % Tag_Name[tag]))
    xmlroot_p = xmltree_p.getroot()

    embedding_mapping = lambda x: embedding_whole(model, x.text, tag)
    embeddings_n = list(map(embedding_mapping, xmlroot_n))
    embeddings_p = list(map(embedding_mapping, xmlroot_p))

    test_idx_n = random.sample(range(len(xmlroot_n)), max(1, int(Test_ratio * len(xmlroot_n))))
    train_idx_n = list(filter(lambda x:x not in test_idx_n, range(len(xmlroot_n))))
    testset_n = list(map(lambda x: embeddings_n[x], test_idx_n))
    testset_n = list(filter(lambda x:x.shape[0] != 0, testset_n))
    trainset_n = list(map(lambda x: embeddings_n[x], train_idx_n))
    trainset_n = list(filter(lambda x:x.shape[0] != 0, trainset_n))

    test_idx_p = random.sample(range(len(xmlroot_p)), max(1, int(Test_ratio * len(xmlroot_p))))
    train_idx_p = list(filter(lambda x:x not in test_idx_p, range(len(xmlroot_p))))
    testset_p = list(map(lambda x: embeddings_p[x], test_idx_p))
    testset_p = list(filter(lambda x:x.shape[0] != 0, testset_p))
    trainset_p = list(map(lambda x: embeddings_p[x], train_idx_p))
    trainset_p = list(filter(lambda x:x.shape[0] != 0, trainset_p))

    trainset_data = trainset_n + trainset_p
    trainset_target = np.array([0] * len(trainset_n) + [1] * len(trainset_p))
    np.savez(os.path.join(Dataset_Dir, Tag_Name[tag], "%s_train.npz" % Tag_Name[tag]), trainset_target , *trainset_data)

    testset_data = testset_n + testset_p
    testset_target = np.array([0] * len(testset_n) + [1] * len(testset_p))
    np.savez(os.path.join(Dataset_Dir, Tag_Name[tag], "%s_test.npz" % Tag_Name[tag]), testset_target , *testset_data)
    

def preprocess():    
    for lan in Languages:
        preprocess_file(os.path.join(Dataset_Dir, Tag_Name[lan], "%s_negative.xml" % Tag_Name[lan]), lan)
        xmltree_n = ET.parse(os.path.join(Dataset_Dir, Tag_Name[lan], "%s_negative.xml" % Tag_Name[lan]))
        preprocess_file(os.path.join(Dataset_Dir, Tag_Name[lan], "%s_positive.xml" % Tag_Name[lan]), lan)
        xmltree_p = ET.parse(os.path.join(Dataset_Dir, Tag_Name[lan], "%s_positive.xml" % Tag_Name[lan]))

        if (not os.path.exists(os.path.join(Dataset_Dir, Tag_Name[lan], "%s_train.npz" % Tag_Name[lan]))) or (not os.path.exists(os.path.join(Dataset_Dir, Tag_Name[lan], "%s_test.npz" % Tag_Name[lan]))):
            print("Pre-calculating the embedding of %s corpus." % str.upper(Tag_Name[lan]))
            cvt_to_npz(lan)
        else:
            print("Embedding of %s corpus detected." % str.upper(Tag_Name[lan]))