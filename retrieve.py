#%%
import pandas as pd
import numpy as np
from os import walk
from BTrees.OOBTree import OOBTree
from collections import deque
import math
import string
import re 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
import time
#from spell_check import *
# import nltk
# nltk.download('stopwords')

#%%
def preprocess(sentence):
    cleaned_sentence = []
    sentence = sentence.lower() # To convert to lowercase
    sentence = re.sub(r'\d+', '', sentence) # To remove numbers

    translator = str.maketrans('', '', string.punctuation)# To remove punctuations
    sentence = sentence.translate(translator)

    sentence = " ".join(sentence.split()) # To strip white spaces

    stop_words = set(stopwords.words("english")) # To remove stopwords
    word_tokens = word_tokenize(sentence) # Tokenize
    cleaned_sentence = [word for word in word_tokens if word not in stop_words]

    stemmer = PorterStemmer() # Applying porter stemmer
    cleaned_sentence = [stemmer.stem(word) for word in cleaned_sentence]

    #cleaned_sentence = ' '.join(cleaned_sentence)
    print('clean:', cleaned_sentence)
    return cleaned_sentence

#%%
# Index for all 417 files together :
files_list = []
for i,j,k in walk("./New_Processed_data/"):
    files_list.extend(k)
# create document file mappings
doc_file_mapping = {pos:val for pos,val in enumerate(files_list)}
# use object-object BTree
standard_inverted_index = OOBTree()
# fill index
row_vector_mapping = {}
rev_row_vector_mapping = {}
no_of_docs = 0
for doc in doc_file_mapping:
    df = pd.read_csv("./New_Processed_data/"+doc_file_mapping[doc])
    # print(doc_file_mapping[doc])
    for row,text in enumerate(df["Snippet"]):
        docID = str(doc)+'_'+str(row)
        row_vector_mapping[docID] = no_of_docs
        rev_row_vector_mapping[no_of_docs] = docID
        no_of_docs += 1
        for pos,term in enumerate(text.split()):
            if standard_inverted_index.has_key(term):
                if docID in standard_inverted_index[term]:
                    standard_inverted_index[term][docID].append(pos)
                else:
                    standard_inverted_index[term][docID] = [pos]
            else:
                standard_inverted_index.update({term:{docID:[pos]}})

#%%
# Index for all 417 files together (Original) :
files_list = []
for i,j,k in walk("./Processed_dataset/"):
    files_list.extend(k)
# create document file mappings
doc_file_mapping = {pos:val for pos,val in enumerate(files_list)}
# use object-object BTree
ori_inverted_index = OOBTree()
# fill index
row_vector_mapping = {}
rev_row_vector_mapping = {}
no_of_docs = 0
for doc in doc_file_mapping:
    df = pd.read_csv("./Processed_dataset/"+doc_file_mapping[doc])
    # print(doc_file_mapping[doc])
    for row,text in enumerate(df["Snippet"]):
        docID = str(doc)+'_'+str(row)
        row_vector_mapping[docID] = no_of_docs
        rev_row_vector_mapping[no_of_docs] = docID
        no_of_docs += 1
        for pos,term in enumerate(text.split()):
            if ori_inverted_index.has_key(term):
                if docID in ori_inverted_index[term]:
                    ori_inverted_index[term][docID].append(pos)
                else:
                    ori_inverted_index[term][docID] = [pos]
            else:
                ori_inverted_index.update({term:{docID:[pos]}})

# %%
# generate permuterm index

files_list = []
for i,j,k in walk("./Processed_dataset/"):
    files_list.extend(k)
# initialise the object-object btree
permuterm_index = OOBTree()
for filename in files_list:
    df = pd.read_csv("./Processed_dataset/"+filename)
    for text in df["Snippet"]:
        for term in text.split():
            # add $ to the end of term as special character to create permuterm
            x = len(term)
            permuterm = term+'$'
            if not permuterm_index.has_key(permuterm):
                # add all permuterms in a dictionary to point to term
                d = {}
                for i in range(x+1):
                    d[permuterm] = term
                    permuterm = permuterm[1:]+permuterm[0]
                permuterm_index.update(d)



#%%
# tf-idf vector calculation

keys = list(standard_inverted_index.keys())
term_index_map = {j:i for i,j in enumerate(keys)}

docs = {}#np.zeros((no_of_docs, len(standard_inverted_index)))
for key,value in standard_inverted_index.items():
    # calculate df
    df = len(value)
    for docID,pos in value.items():
        tf = len(pos)
        idf = math.log10(no_of_docs/df)
        tf_idf = tf * idf
        # filling up the tf-idf vectors (docs)
        if row_vector_mapping[docID] in docs:
            docs[row_vector_mapping[docID]][key] = tf_idf
        else:
            docs[row_vector_mapping[docID]] = {key : tf_idf}
        #print(key , "TF-IDF Score is : " , tf_idf)


    # a set containing all unique files with the key
    """ file_arr = set()
    no_of_occurrances = 0
    for new_key,new_value in value.items():
        temp_arr = new_key.split("_")
        file_arr.add(int(temp_arr[0]))
        no_of_occurrances = no_of_occurrances + len(new_value)
    no_of_files = len(file_arr)
    tf = 1 + math.log10(no_of_occurrances)
    idf = math.log10(417/no_of_files)
    tf_idf = tf * idf
    # filling up the tf-idf vectors (D)
    ind = keys.index(key)
    # since we have all the file ids with the term we can directly update the tf-idf vectors (D)
    # -1 because file ids start from 1 to 417
    for doc in file_arr:
        D[doc-1][ind] = tf_idf
    #print(key , "TF-IDF Score is : " , tf_idf)
    standard_inverted_index[key]["tf-idf"] = tf_idf """

# %%
# wildcard queries
def query_func(perm_tree, index_tree, query):
    splits = query.split('*')
    # for *X*, finding X*
    if query[0] == query[-1] == '*':
        min_query = splits[1]
        max_query = splits[1][:-1] + chr(ord(splits[1][-1])+1)
        words = list(permuterm_index.items(min_query, max_query))
    # for *X, finding X$*
    elif query[0] == '*':
        min_query = splits[1] + '$'
        max_query = splits[1] + 'z' #value needs updating
        # print(splits,max_query)
        words = list(permuterm_index.items(min_query, max_query))
    # for X*, finding $X*
    elif query[-1] == '*':
        min_query = '$'+ splits[0]
        max_query = '$' + splits[0][:-1] + chr(ord(splits[0][-1])+1)
        words = list(permuterm_index.items(min_query, max_query))
    # for X*Y, finding Y$X*
    else:
        min_query = splits[1] + '$' + splits[0]
        max_query = splits[1] + '$' + splits[0][:-1] + chr(ord(splits[0][-1])+1)
        words = list(permuterm_index.items(min_query, max_query))
    list_of_words = list(map(lambda x: x[1],words))
    # for i in words:
    #     print(i[1], 'in', index_tree[i[1]])
    return list_of_words


#%%
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

#%%
def gen_vector(tokens):

    Q = np.zeros((len(standard_inverted_index)))
    
    counter = Counter(tokens)
    
    for token in np.unique(tokens):
        # print("1",token)
        tf = counter[token]
        df = len(standard_inverted_index[token])
        idf = math.log10(no_of_docs/df)
        if(token in standard_inverted_index):
            ind = term_index_map[token]
            Q[ind] = tf * idf
            print(token)
        else:
            pass

    return Q

#%%
def gen_vec(query,each_doc):
    keys = docs[each_doc].keys()
    tokens = set(query) | set(keys)
    query_vector = np.zeros((len(tokens)))
    doc_vector = np.zeros((len(tokens)))
    l_tokens = list(sorted(tokens))

    counter = Counter(tokens)
        
    for i,j in enumerate(l_tokens):
        if j in query:
            tf = counter[j]
            df = len(standard_inverted_index[j])
            idf = math.log10(no_of_docs/df)
            query_vector[i] = tf * idf
        if j in keys:
            doc_vector[i] = docs[each_doc][j]

    return query_vector,doc_vector


#%%
def cosine_similarity(k, tokens):
    print("Cosine Similarity")
    
    print("\nQuery:", corrected_query)
    print("")
    #print(tokens)
    
    d_cosines = np.zeros((no_of_docs))
    
    """ query_vector = gen_vector(tokens)
    print('query_vec:',query_vector)
    
    for each_doc in docs: #10^5
        doc_vec = np.zeros((len(standard_inverted_index)))
        for term in docs[each_doc]: #10 ^ 1
            ind = term_index_map[term] #10^
            doc_vec[ind] = docs[each_doc][term]
        d_cosines[each_doc] = cosine_sim(query_vector, doc_vec) """
    
        
    for each_doc in docs:
        query_vector,doc_vec = gen_vec(tokens,each_doc)
        d_cosines[each_doc] = cosine_sim(query_vector, doc_vec)
   

    out = np.array(d_cosines).argsort()[-k:][::-1]
    scores = sorted(np.array(d_cosines))[-k:][::-1]
    print("")
    
    print(out)
    for i,j in zip(out,scores):
        # print("line : ", i)
        doc,row = map(int,rev_row_vector_mapping[i].split('_'))
        print("Filename : " , doc_file_mapping[doc],doc,row,j)

#%%

start_time = time.time()
query = input("Enter your Query")
if '*' not in query:        
    # corrected_query = spell_correct_context(query)
    # print(corrected_query)
    corrected_query = query
    tokens = preprocess(corrected_query)
    cosine_similarity(10,tokens)
else:
    res = query_func(permuterm_index, ori_inverted_index, query)
    print(res)

end_time = time.time()
print('time:',end_time-start_time)



# %%
