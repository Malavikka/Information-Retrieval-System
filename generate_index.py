#%%
# inverted index implementation details : 
# The dictionary will be a B-Tree  [Other options: Trie, Hash-Table(dictionary)]
#   The key will be the term and the value will be postings list
# The postings list will be an array(list) [Other options: Linked List]
#   Posting list will be a positional index [i.e., list of docIDs with the positions of the occurance of the term in that doc]

# %%
# importing required modules
import pandas as pd
import numpy as np
from os import walk
from BTrees.OOBTree import OOBTree
from collections import deque
import math

# %%
# BTree module usage:
# to add key,value into BTree, use tree.update({key1:value1,key2:value2})
# to check if term present in tree use, tree.has_key(term), returns true if present, else false if not

# %%
# our index structure: 
# BTree: key = term, value = dictionary [key = docID, value = list with positions of occurance of term]
# docID = docNo_rowNo   [docNo = csv no]

# %%
# For individual csv files : 
filename = "BBCNEWS.201701.csv"
index = OOBTree()
print(index)
df = pd.read_csv("./Processed_dataset/"+filename)
doc = "1"
for row,text in enumerate(df["Snippet"]):
    docID = doc+'_'+str(row)
    for pos,term in enumerate(text.split()):
        if index.has_key(term):
            if docID in index[term]:
                index[term][docID].append(pos)
            else:
                index[term][docID] = [pos]
        else:
            index.update({term:{docID:[pos]}})

# %%
# To display index :
x = list(index.items())
x

# %%
# Index for all 417 files together :
my_inverted_index = dict() #
files_list = []
for i,j,k in walk("./Processed_dataset/"):
    files_list.extend(k)
# create document file mappings
doc_file_mapping = {pos:val for pos,val in enumerate(files_list)}
# use object-object BTree
standard_inverted_index = OOBTree()
# fill index
for doc in doc_file_mapping:
    df = pd.read_csv("./Processed_dataset/"+doc_file_mapping[doc])
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
# Cell to calculate the TF-IDF scores.
# Term Frequency = 1 + log to the base 10 of number of occurances of the term in a document
# Inverse Document Frequency = log to the base 10 of (N/df)  

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

normalize_the_score(standard_inverted_index,my_inverted_index)

#%%

z = list(standard_inverted_index.items())
z

#vocab = list()
#for key,value in standard_inverted_index.items():
#    vocab.append(key)
#vocab

# %%
# Permuterm index : 
#   For each term we add $ and rotate it to create permuterms, and add this into a BTree
#   Key = Permuterm, value = original term
#   Eg : term = hello | permuterms = hello$,ello$h,llo$he,lo$hel,o$hell

# %%
# for a single document
# initialise the object-object btree
permuterm_index = OOBTree()
filename = "BBCNEWS.201701.csv"
df = pd.read_csv("./Processed_dataset/"+filename)
for text in df["Snippet"]:
    for term in text.split():
        # add $ to the end of term as special character to create permuterm
        x = len(term)
        permuterm = term+'$'
        if not permuterm_index.has_key(permuterm):
            # add all permuterms in a dictionary to point to term
            d = {}
            # method 1 : list splicing
            for i in range(x):
                d[permuterm] = term
                permuterm = permuterm[1:]+permuterm[0]
            # method 2 : deque rotations 
            # dq = deque(permuterm)
            # for i in range(x):
            #     d[''.join(dq)]=term
            #     dq.rotate(1)
            #     if i==0:
            #         dq.rotate(1)
            # fill in the index by adding the permuterm-term mapping
            permuterm_index.update(d)

# %%
# for all 418 documents
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
                # method 1 of generating permuterms
                for i in range(x):
                    d[permuterm] = term
                    permuterm = permuterm[1:]+permuterm[0]
                # method 2 of generating permuterms
                # dq = deque(permuterm)
                # for i in range(x):
                #     d[''.join(dq)] = term
                #     dq.rotate(1)
                #     if i==0:
                #         dq.rotate(1)
                # fill in the index by adding the permuterm-term mapping
                permuterm_index.update(d)

# %%
x = list(permuterm_index.items())
x
    
# %%
# todo : time the method 1 and 2 of permuterm generation and see which is faster... By general overview, both seem to more or less take the same time.
# %%
