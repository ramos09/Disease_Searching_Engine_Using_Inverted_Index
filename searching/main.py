import numpy as np
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from itertools import count
import collections
import math
from math import log


import xml.dom.minidom as minidom
collection = minidom.parse("searching/data.xml")

doc_no = collection.getElementsByTagName('sym')
gejala = collection.getElementsByTagName('symptom')
penyakit = collection.getElementsByTagName('disease')


N_DOC = len(doc_no)

doc_gejala = []
for i in range(N_DOC):
    sentence = gejala[i].firstChild.data
    doc_gejala.append(sentence)

doc_number = []
for i in range(N_DOC):
    number = doc_no[i].firstChild.data
    doc_number.append(number)

doc_penyakit = []
for i in range(N_DOC):
    title = penyakit[i].firstChild.data
    doc_penyakit.append(title)



all_text = []
for i in range(len(doc_number)):
    all_text.append(doc_gejala[i]+doc_penyakit[i])


#Tokenisasi
def remove_punc_tokenize(sentence):
    tokens = []

    for w in CountVectorizer().build_tokenizer()(sentence):
        tokens.append(w)
    return tokens

tokens_doc = []
for i in range(N_DOC):
    tokens_doc.append(remove_punc_tokenize(all_text[i]))

tokens_doc

#Case Folding
def to_lower(tokens):
    tokens = [x.lower() for x in tokens]
    return tokens

for i in range(N_DOC):
    tokens_doc[i] = to_lower(tokens_doc[i])


tokens_doc


#Stopping
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def stop_word_token(tokens):
    tokens = [w for w in tokens if not w in stop_words]
    return tokens

for i in range(N_DOC):
    tokens_doc[i] = stop_word_token(tokens_doc[i])



#Normalization
stemmer = PorterStemmer()

def stemming(tokens):
    for i in range(0, len(tokens)):
        if (tokens[i] != stemmer.stem(tokens[i])):
            tokens[i] = stemmer.stem(tokens[i])
    return tokens

for i in range(N_DOC):
    tokens_doc[i] = stemming(tokens_doc[i])


#Proximity Index
all_tokens = []

for i in range(N_DOC):
    for w in tokens_doc[i]:
        all_tokens.append(w)

new_sentence = ' '.join([w for w in all_tokens])

all_tokens = set(all_tokens)

try:
    from itertools import izip as zip
except ImportError:
    pass
# proximity_index = {}

# for token in all_tokens:
#     dict_doc_position = {}
#     for n in range(N_DOC):
#         if(token in tokens_doc[n]):
#             dict_doc_position[doc_no[n].firstChild.data] = [i for i, j in zip(count(), tokens_doc[n]) if j == token]
#     proximity_index[token] = dict_doc_position


# proximity_index = collections.OrderedDict(sorted(proximity_index.items()))
inverted_index = {}
for doc_id in range(len(all_tokens)):
    for term in tokens_doc[doc_id]:
        if term not in inverted_index:
            inverted_index[term] = []
        inverted_index[term].append(doc_id) 




def main(query):
    list_of_query = [query.split()]

    # Stopwords Process
    for i in range(len(list_of_query)):
        list_of_query[i] = [w for w in list_of_query[i] if not w in stopwords.words('english')]

    # Case Folding Process
    for i in range(len(list_of_query)):
        list_of_query[i] = [kata.lower() for kata in list_of_query[i]]

    # Stemming Process
    for i in range(len(list_of_query)):
        list_of_query[i] = [stemmer.stem(kata) if kata!=stemmer.stem(kata) else kata for kata in list_of_query[i]]

    queries = []
    for i in range(len(list_of_query)):
        for kata in list_of_query[i]:
            if not kata in queries:
                 queries.append(kata)

#TFIDF
    N = len(tokens_doc)
    df = []
    res = []

    for i in range(len(queries)):
        sums = 0
        for j in range(len(tokens_doc)):
            if queries[i] in tokens_doc[j]:
                sums += 1
        df.append(sums)


    for i in range(len(df)):
        if df[i] != 0:
            res.append(math.log10(N / df[i]))
        else:
            res.append(0)


    weight = []

    for i in range(len(queries)):
        lists = []
        for j in range(len(tokens_doc)):
            dicts = {}
            x = tokens_doc[j].count(queries[i])
            if x == 0:
                dicts[j+1] = 0
                lists.append(dicts)
            else:
                score = math.log10(x)
                score += 1
                score *= res[i]
                dicts[j+1] = score
                lists.append(dicts)
        weight.append(lists)


    weight

    result = []
    for i in range(len(list_of_query)):
        l = []
        for j in range(len(tokens_doc)):
            dic = {}
            for kata in list_of_query[i]:
                sums = 0
                ind = queries.index(kata)
                #print(ind)
                for val in weight[ind][j].values():
                    sums += val
            if(sums!= 0):
                dic['docno'] = j+1
                dic['score'] = sums
                dic['penyakit'] = doc_penyakit[j]
                dic['gejala'] = doc_gejala[j]
    #             dic['text'] = doc_text[j]
            if(len(dic) != 0): l.append(dic)
        result.append(l)

    result



    for i in range(len(list_of_query)):
        result[i] = sorted(result[i], key = lambda x : x['score'], reverse = True)

    for i in range(len(list_of_query)):
        with open('resultquery.txt'.format(counter = i+1), 'w') as f:
            f.write('Top 5 Documents :\n')
            f.write('q_Id - DOC NO - Pekerjaan - SCORE\n')
            if len(result[i]) > 5:
                for x in range(5):
                    c = i + 1
                    f.write('%s   -   %s   -   %s   -   %s\n' %(c,doc_number[result[i][x]['docno']-1],result[i][x]['gejala'],result[i][x]['score']))
            else:
                for x in result[i]:
                    c  = i + 1
                    f.write('%s   -   %s   -   %s    -   %s\n' %(c,doc_number[x['docno']-1],x['gejala'],x['score']))
#Cosine Similarity
    freq = []
    for i in range(len(queries)):
        s = 0
        for x in range(len(list_of_query)):
            if queries[i] in list_of_query[x]:
                s += 1
        freq.append(s)

    resultqueries = []
    for i in range(len(freq)):
        resultqueries.append(math.log1p(N / freq[i]))


    weightqueries = []
    for i in range(len(queries)):
        lists = []
        for j in range(len(list_of_query)):
            dicts = {}
            x = list_of_query[j].count(queries[i])
            if x == 0:
                dicts[j+1] = 0
                lists.append(dicts)
            else:
                score = math.log10(x)
                score += 1
                score *= resultqueries[i]
                dicts[j+1] = score
                lists.append(dicts)
        weightqueries.append(lists)



    new_weight = []

    for i in range(len(queries)):
        lists = []
        for j in range(len(tokens_doc)):
            dicts = {}
            x = tokens_doc[j].count(queries[i])
            if x == 0:
                dicts[j+1] = 0
                lists.append(dicts)
            else:
                score = math.log10(x)
                score += 1
                dicts[j+1] = score
                lists.append(dicts)
        new_weight.append(lists)


    normalize = []
    for i in range(len(queries)):
        ss = []
        g = 0
        for x in weightqueries[i]:
            for val in x.values():
                if val!=0: ss.append(val)
        for c in ss:
            g = g + math.pow(c,2)
        normalize.append(math.sqrt(g))


    for i in range(len(queries)):
        for x in weightqueries[i]:
            for key,val in x.items():
                if normalize[i] != 0:
                    val = val / normalize[i]
                    x[key] = val
                else:
                    res.append(0)


    length2 = len(tokens_doc)
    normalization = []
    for i in range(len(queries)):
        ss = []
        g = 0
        for x in new_weight[i]:
            for val in x.values():
                if val != 0: ss.append(val)
        for c in ss:
            g = g + math.pow(c,2)
        normalization.append(math.sqrt(g))


    for i in range(len(queries)):
        for x in new_weight[i]:
            for key,val in x.items():
                if normalization[i] != 0:
                    val = val / normalization[i]
                    x[key] = val
                else:
                    res.append(0)


    result_cosine = []
    for i in range(len(list_of_query)):
        hasilcosine  = []
        for j in range(len(tokens_doc)):
            dix  = {}
            ans = []
            for kata in list_of_query[i]:
                ind = queries.index(kata)
                for x,y in zip(weightqueries[ind][i].values(),new_weight[ind][j].values()):
                    ans.append(x*y)

            if sum(ans)!=0:
                dix['docno'] = j+1
                dix['score'] = sum(ans)
                dix['penyakit'] = doc_penyakit[j]
                dix['gejala'] = doc_gejala[j]
            if len(dix) != 0: hasilcosine.append(dix)
        result_cosine.append(hasilcosine)


    xx = result_cosine
    for i in range(len(list_of_query)):
        result_cosine[i] = sorted(result_cosine[i], key = lambda x : x['score'], reverse = True)

    result_cosine[0]

    top_res = result_cosine[0]
    top_result = top_res[:50]
    resultatas = top_result[:1]

    return top_result,resultatas,all_tokens, tokens_doc, query, queries,weight,inverted_index
