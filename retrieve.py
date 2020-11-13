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
from spell_check import *
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
    return cleaned_sentence

#%%
# Index for all 417 files together :
my_inverted_index = dict() #
files_list = []
for i,j,k in walk("./New_Processed_data/"):
    files_list.extend(k)
# create document file mappings
doc_file_mapping = {pos:val for pos,val in enumerate(files_list)}
# use object-object BTree
standard_inverted_index = OOBTree()
# fill index
for doc in doc_file_mapping:
    df = pd.read_csv("./New_Processed_data/"+doc_file_mapping[doc])
    print(doc_file_mapping[doc])
    for row,text in enumerate(df["Snippet"]):
        if(doc not in my_inverted_index): #
            my_inverted_index[doc] = list() #
        docID = str(doc)+'_'+str(row)
        for pos,term in enumerate(text.split()):
            if standard_inverted_index.has_key(term):
                if docID in standard_inverted_index[term]:
                    standard_inverted_index[term][docID].append(pos)
                else:
                    standard_inverted_index[term][docID] = [pos]
            else:
                standard_inverted_index.update({term:{docID:[pos]}})
                my_inverted_index[doc].append(term) #

#%%
def find_no_of_files(arr):
    s = set(arr)
    return len(s)

def normalize_the_score(standard_inverted_index,my_inverted_index):
    for key,value in my_inverted_index.items():
        denominator = 0
        for term in value:
            term_TfIdf = standard_inverted_index[term]["tf-idf"]
            term_TfIdf = math.pow(term_TfIdf,2)
            denominator = denominator + term_TfIdf
        normalized_denominator = math.sqrt(denominator)
        for term in value:
            standard_inverted_index[term]["tf-idf"] = standard_inverted_index[term]["tf-idf"]/normalized_denominator
            standard_inverted_index[term]["tf-idf"] = round(standard_inverted_index[term]["tf-idf"],5)

for key,value in standard_inverted_index.items():
    file_arr = list()
    no_of_occurrances = 0
    for new_key,new_value in value.items():
        temp_arr = new_key.split("_")
        file_arr.append(int(temp_arr[0]))
        no_of_occurrances = no_of_occurrances + len(new_value)
    no_of_files = find_no_of_files(file_arr)
    tf = 1 + math.log10(no_of_occurrances)
    idf = math.log10(417/no_of_files)
    tf_idf = tf * idf
    #print(key , "TF-IDF Score is : " , tf_idf)
    standard_inverted_index[key]["tf-idf"] = tf_idf

#normalize_the_score(standard_inverted_index,my_inverted_index)

#%%
standard_inverted_index = list(standard_inverted_index.items())
standard_inverted_index

#%%
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

#%%
D = np.zeros((417, len(standard_inverted_index)))
for i in standard_inverted_index:
    try:
        ind = standard_inverted_index.index(i)
        D[i[0]][ind] = standard_inverted_index[i]["tf-idf"]
    except:
        pass

#%%
def gen_vector(tokens):

    Q = np.zeros((len(standard_inverted_index)))
    
    counter = Counter(tokens)
    words_count = len(tokens)
    
    for token in np.unique(tokens):
        
        #tf = counter[token]/words_count
        #df = doc_freq(token)
        #idf = math.log((N+1)/(df+1))
        if(token in standard_inverted_index):
            ind = standard_inverted_index.index(token)
            Q[ind] = standard_inverted_index[token]["tf-idf"]
        else:
            pass

    return Q
#%%
def cosine_similarity(k, tokens):
    print("Cosine Similarity")
    
    print("\nQuery:", corrected_query)
    print("")
    #print(tokens)
    
    d_cosines = []
    
    query_vector = gen_vector(tokens)
    
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))
        
    out = np.array(d_cosines).argsort()[-k:][::-1]
    
    print("")
    
    print(out)
    for i in out:
        print("Filename : " , doc_file_mapping[i])

#%%

query = input("Enter your Query")
corrected_query = spell_correct_context(query)
# print(corrected_query)
tokens = preprocess(corrected_query)
Q = cosine_similarity(10,tokens)


# %%