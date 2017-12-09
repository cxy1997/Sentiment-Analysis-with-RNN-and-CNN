from __future__ import print_function, division
import os
import re
import numpy as np
from constants import *
from word_embedding import load_word2vec, embedding
try:
    import xml.etree.cElementTree as ET 
except ImportError:
    import xml.etree.ElementTree as ET

def preprocess_string(s):
    if s[-1] == '\n':
        s = s[:-1]
    return re.sub(r'&(?!(lt;)|(gt;)|(amp;)|(apos;)|(quot;))', '&amp;', s)
    
def preprocess_file(fname):
    with open(fname, 'r') as f_in:
        lines = f_in.readlines()
        lines = list(filter(lambda x:x != '\n', lines))
        for i in range(len(lines)):
            lines[i] = preprocess_string(lines[i])
            
    with open(fname, 'w') as f_out:
        for line in lines:
            f_out.write(line + '\n')

def cvt_to_npz(tag):
    model = load_word2vec(tag)
    xmltree_n = ET.parse(os.path.join(Dataset_Dir, Tag_Name[tag], "%s_negative.xml" % Tag_Name[tag]))
    xmlroot_n = xmltree_n.getroot()
    xmltree_p = ET.parse(os.path.join(Dataset_Dir, Tag_Name[tag], "%s_positive.xml" % Tag_Name[tag]))
    xmlroot_p = xmltree_p.getroot()
    mapping = lambda x: embedding(model, x.text, tag)
    embeddings = list(map(mapping, xmlroot_n)) + list(map(mapping, xmlroot_p))
    target = np.array([0] * len(xmlroot_n) + [1] * len(xmlroot_p))
    np.savez(os.path.join(Dataset_Dir, Tag_Name[tag], "%s_embedding.npz" % Tag_Name[tag]), target, *embeddings)

if __name__ == '__main__':
    '''
    datasets = ["dataset/cn/cn_positive.xml", "dataset/cn/cn_negative.xml", "dataset/en/en_positive.xml", "dataset/en/en_negative.xml", "dataset/task2input.xml"]
    for dataset in datasets:
        preprocess_file(dataset)
    '''
    for lan in Languages:
        cvt_to_npz(lan)