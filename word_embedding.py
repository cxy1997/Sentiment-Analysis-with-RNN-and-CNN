#coding:utf-8
import os
import gensim
import numpy as np
import nltk
import jieba
import re

Word_Embedding_Dir = 'word_embedding'
if not os.path.exists(os.path.join(os.environ['HOME'], 'nltk_data/tokenizers/punkt')):
    nltk.download('punkt')

CN = 0
EN = 1
Embedding_Dim = [256, 100]

def load_word2vec(Tag):
    if Tag == CN:
        return gensim.models.Word2Vec.load(os.path.join(Word_Embedding_Dir, 'word2vec_wx')) # Embedding_Dim = 256
    elif Tag == EN:
        return gensim.models.Word2Vec.load(os.path.join(Word_Embedding_Dir, 'en_word2vec.txt')) # Embedding_Dim = 100

def div_en_sen(text):
    return nltk.sent_tokenize(text)

def div_en_word(sen):
    return nltk.word_tokenize(sen)

def div_cn_sen(text):
    #return re.split("。|(？)|(！)|(？！)|(！？)|\n", text)
    tmp = re.split("(。|？|！|\n)", text)
    pun_save = ['？', '！', '。']
    pun_del = ['，']
    while '' in tmp:
        tmp.remove('')
    length = len(tmp)
    i = 1
    while i < length:
        if tmp[i] in pun_save:
            while tmp[i] in pun_save:
                tmp[i - 1] += tmp[i]
                length -= 1
                del tmp[i]
        elif tmp[i] in pun_del:
            length -= 1
            del tmp[i]
        else:
            i += 1

    return tmp

def div_cn_word(sen):
    return jieba.lcut(sen)

def fix_nltk_words(words):
    for i in range(len(words)):
        if words[i] == '\'ve':
            words[i] = 'have'
        elif words[i] == '\'m':
            words[i] = 'am'
        elif words[i] == '\'d':
            words[i] = 'had'
        elif words[i] == 'n\'t':
            words[i] = 'not'
    return words

def div_sentence(text, Tag):
    if Tag == CN:
        sentences = div_cn_sen(text)
    elif Tag == EN:
        sentences = div_en_sen(text)
    while '' in sentences:
        sentences.remove('')
    while '\n' in sentences:
        sentences.remove('\n')
    return sentences

def div_word(sentence, Tag):
    if Tag == CN:
        return div_cn_word(sentence)
    elif Tag == EN:
        return fix_nltk_words(div_en_word(sentence))

def embedding(model, text, Tag, maxlen = 300):
    sentences = div_sentence(text, Tag)
    word_embedding_matrix = np.zeros((len(sentences), maxlen, Embedding_Dim[Tag]))
    for i, sentence in enumerate(sentences):
        words = div_word(sentence, Tag)
        print(words)
        for j, word in enumerate(words):
            try:
                word_embedding_matrix[i][j] = model[word]
            except KeyError:
                word_embedding_matrix[i][j] = 0.0
    return word_embedding_matrix

if __name__ == '__main__':
    # model_en = load_word2vec(EN)
    model_cn = load_word2vec(CN)
    #print(model.wv['reinforcement'])
    #print(model.most_similar(positive=['good'],topn=10))
    # text_en = "Yes, this product does exactly what it says it does and it does it very well - better than any other i've tried...BUT it doesn't work as well on all fur types. I have one terrier mix with fluffy fur/hair and two pugs. The Furminator works wonders on the terrier mix, on my male pug it works okay and on my female pug it pulls nothing (perhaps someone can make a suggestion?). It is worth my expense only because completely deshedding just one dog helps."
    text_cn = "我从十二岁起！便在镇口的咸亨酒店里当伙计？\n掌柜说！？样子太傻？！外面的短衣主顾。虽然容易说话，但唠唠叨叨缠夹不清的也很不少。他们往往要亲眼看着黄酒从坛子里舀出，看过壶子底里有水没有，又亲看将壶子放在热水里，然后放心：在这严重监督下，羼水也很为难。所以过了几天，掌柜又说我干不了这事。幸亏荐头的情面大，辞退不得，便改为专管温酒的一种无聊职务了。\n"
    # print(embedding(model_en, text_en, EN).shape)
    print(embedding(model_cn, text_cn, CN).shape)
    print(model_cn['十二岁'])
    print(model_cn['情面'])
    print(model_cn['掌柜'])
