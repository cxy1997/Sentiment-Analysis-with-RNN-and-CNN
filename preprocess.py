from __future__ import print_function, division
import re

def preprocess_string(s):
    if s[-1] == '\n':
        s = s[:-1]
    return re.sub(r'&(?!(lt;)|(gt;)|(amp;)|(apos;)|(quot;))', '&amp;', s)
    
def preprocess_file(fname):
    with open(fname, 'r') as f_in:
        lines = f_in.readlines()
        lines = filter(lambda x:x != '\n',lines)
        for i in range(len(lines)):
            lines[i] = preprocess_string(lines[i])
            
    with open(fname, 'w') as f_out:
        for line in lines:
            f_out.write(line + '\n')

if __name__ == '__main__':
    datasets = ["dataset/cn/cn_positive.xml", "dataset/cn/cn_negative.xml", "dataset/en/en_positive.xml", "dataset/en/en_negative.xml", "dataset/task2input.xml"]
    for dataset in datasets:
        preprocess_file(dataset)
