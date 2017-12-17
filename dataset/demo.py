#coding:utf-8
from __future__ import print_function, division
import os
import re
import numpy as np
import random
try:
    import xml.etree.cElementTree as ET 
except ImportError:
    import xml.etree.ElementTree as ET

xmltree = ET.parse('task2output.xml')
xmlroot = xmltree.getroot()
for review in xmlroot:
    print(review.text)
    if int(review.attrib['polarity']) > 0:
        print('\033[1;32m' + '好评' + '\033[0m')
    else:
        print('\033[1;31m' + '差评' + '\033[0m')
    print()