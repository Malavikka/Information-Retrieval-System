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
import requests
from spell_check import *
# import nltk
# nltk.download('stopwords')



#%%
# Index for all 417 files together :
files_list = []
for i,j,k in walk("./New_Processed_data/"):
    files_list.extend(k)
# create document file mappings
doc_file_mapping = {pos:val for pos,val in enumerate(files_list)}
# use object-object BTree
standard_inverted_index = OOBTree()
# row and vector mappings
row_vector_mapping = {}
rev_row_vector_mapping = {}
# URL doc_row mappings
url_map = {}
for doc in doc_file_mapping:
    df = pd.read_csv("./Dataset/"+doc_file_mapping[doc])
    for row,url in enumerate(df["URL"]):
        docID = str(doc)+'_'+str(row)
        url_map[docID] = url
# fill index
no_of_docs = 0
for doc in doc_file_mapping:
    df = pd.read_csv("./New_Processed_data/"+doc_file_mapping[doc])
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

docs = {}
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


#%%
def preprocess(sentence):
    cleaned_sentence = []
    sentence = sentence.lower() # To convert to lowercase
    sentence = re.sub(r'\d+', '', sentence) # To remove numbers

    translator = str.maketrans('', '', string.punctuation) # To remove punctuations
    sentence = sentence.translate(translator)

    sentence = " ".join(sentence.split()) # To strip white spaces

    stop_words = set(stopwords.words("english")) # To remove stopwords
    word_tokens = word_tokenize(sentence) # Tokenize
    cleaned_sentence = [word for word in word_tokens if word not in stop_words]

    stemmer = PorterStemmer() # Applying porter stemmer
    cleaned_sentence = [stemmer.stem(word) for word in cleaned_sentence]

    return cleaned_sentence


#%%
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


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
            if standard_inverted_index.has_key(j):
                df = len(standard_inverted_index[j])
                idf = math.log10(no_of_docs/df)
            else:
                idf = 0
            query_vector[i] = tf * idf
        if j in keys:
            doc_vector[i] = docs[each_doc][j]

    return query_vector,doc_vector

#%%

def get_elastic_search_results(orig_query):
    url = "http://localhost:9200/_search?q="
    mod_query = "+".join(orig_query.split())
    url = url+mod_query
    url1 = url+'&size=0&track_total_hits=true'
    size = requests.get(url1).json()["hits"]["total"]["value"]
    url = url+'&size='+str(size)
    print("\nElastic Search request URL : ",url)
    start = time.time()
    response = requests.get(url).json()
    end = time.time()
    es_res = set()
    f = open("es_res.txt",'w')
    ctr = 0
    for hit in response["hits"]["hits"]:
        if ctr<100:
            print("Filename : " ,hit["_index"].upper()+".csv",file=f)
            print("Score : ",hit["_score"],file=f)
            for i,j in hit["_source"].items():
                print(i," : ",j,file=f)
        es_res.add(hit["_source"]["URL"])
        if ctr<100:
            print("-----",file=f)
            ctr+=1
    f.close()
    es_time = (end-start,response["took"])
    return es_res,es_time

#%%

def display_metrics(out,scores,d_cosines,orig_query):
    ir_res = set()
    # print(out)
    # display top 10 to std_out
    print("\nTop 15 results : [Ranked on cosine similarity scores]")
    for i,j in zip(out[:15],scores[:15]):
        print("-----")
        doc,row = map(int,rev_row_vector_mapping[i].split('_'))
        print("Document Name : " , doc_file_mapping[doc],"\nRow Number : ",row,"\nCosine Similarity Score : ",j)
        df = pd.read_csv('./Dataset/'+doc_file_mapping[doc])
        print(df.iloc[row]["Snippet"])
        print("-----")
    x = 0
    while (scores[x]>0.001):
        x+=1
    out = out[:x]
    scores = scores[:x]
    # fill top 100 hits into ir_res.txt and populate ir_res set with URLs
    f = open('ir_res.txt','w')
    ctr = 0
    for i,j in zip(out,scores):
        doc,row = map(int,rev_row_vector_mapping[i].split('_'))
        ir_res.add(url_map[str(doc)+'_'+str(row)])
        if ctr<100:
            print("Filename : " , doc_file_mapping[doc],"\nrow_number : ",row,"\ncosine score : ",j,file=f)
            df = pd.read_csv('./Dataset/'+doc_file_mapping[doc])
            for i,j in df.iloc[row].items():
                print(i,' : ',j,file=f)
            print('--------',file=f)
            ctr+=1
    # call get_elastic_search_results to get urls of top 100 elastic search results
    es_res,es_time = get_elastic_search_results(orig_query)
    print("\n-----\nMetrics with comparison to elastic search : ")
    # True Positive is all the files retrieved by both
    TP = len(ir_res.intersection(es_res))
    # False Positive are files retrieved by Our IR and those not retrieved by elastic search
    FP = len(ir_res-es_res)
    # False Negative are those retrieved by Elastic but missed by our algorithm
    FN = len(es_res-ir_res)
    # True negative are those that are not retrieved by elasticsearch or our algorithm
    TN = len(d_cosines) - (TP+FP+FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    fscore = (2*precision*recall)/(precision+recall)
    print("\nAccuracy : ",accuracy,"\n\nPrecision : ",precision,"\n\nRecall : ",recall,"\n\nF-Score : ",fscore)
    print("\nConfusion Matrix :[[TP,FP][FN,TN]]")
    print([[TP,FP],[FN,TN]])
    print("\n-----\nTiming : \n")
    print("Elastic Search Time : (timed api, value returned by api)",es_time)



#%%
def spell_correct_context(query_str):
    corrector = jamspell.TSpellCorrector()    # Create a corrector
    corrector.LoadLangModel('./en.bin')  
    # list_of_words = query_str.split()
    #PRINTING THE CANDIDATES 
    # for i in range(len(list_of_words)):
    #     print(list_of_words[i]+" -> ", corrector.GetCandidates(list_of_words, i))
    # print("Did you mean " + "'"+corrector.FixFragment(query_str)+ "'"+"?")
    return corrector.FixFragment(query_str)

#%%
def regular_query(k, query):
    start_time = time.time()
    # corrected_query = query

    corrected_query = spell_correct_context(query)
    print('corrected_query:',corrected_query)
    tokens = preprocess(corrected_query)
    
    print("\nQuery:", corrected_query)
    print("")

    d_cosines = np.zeros((no_of_docs))

    for each_doc in docs:
        query_vector,doc_vec = gen_vec(tokens,each_doc)
        d_cosines[each_doc] = cosine_sim(query_vector, doc_vec)    
   
    out = np.array(d_cosines).argsort()[::-1]
    scores = sorted(np.array(d_cosines))[::-1]
    print("")
    end_time = time.time()
    print("Retrieval Time : ",end_time-start_time)
    display_metrics(out,scores,d_cosines,query)
    print("\nOur Retrieval Time : ",end_time-start_time)


#%%
#phrase query
def phrase_query(tokens, standard_inverted_index):
    result = []
    docs = []
    for i in tokens:
        if standard_inverted_index.has_key(i):
            docsTerm = []
            postings = standard_inverted_index[i]
            for j in postings:
                docsTerm.append(j)
            docs.append(docsTerm)
    setted = set(docs[0]).intersection(*docs)

    for filename in setted:
        temp = []
        for i in tokens:
            postingList = standard_inverted_index[i][filename]
            temp.append(postingList)

        for i in range(len(temp)):
            for ind in range(len(temp[i])):
                temp[i][ind] -= i
        if set(temp[0]).intersection(*temp):
            result.append(filename)
    return result

#%%
def ranked_phrase_query(query):
    start_time = time.time()
    corrected_query = spell_correct_context(query)
    tokens = preprocess(corrected_query)
    result = phrase_query(tokens, standard_inverted_index)

    d_cosines = np.zeros((no_of_docs))

    for each_docID in result:
        query_vector,doc_vec = gen_vec(tokens,row_vector_mapping[each_docID])
        d_cosines[row_vector_mapping[each_docID]] = cosine_sim(query_vector, doc_vec)

    out = np.array(d_cosines).argsort()[::-1]
    scores = sorted(np.array(d_cosines))[::-1]
    print("")
    end_time = time.time()
    print("Retrieval Time : ",end_time-start_time)
    display_metrics(out,scores,d_cosines,query)
    print("\nOur Retrieval Time : ",end_time-start_time)

# %%
# wildcard queries
def wildcard_query(query):
    start_time = time.time()
    splits = query.split('*')
    # for *X*, finding X* 
    if query[0] == query[-1] == '*':
        min_query = splits[1]
        max_query = splits[1][:-1] + chr(ord(splits[1][-1])+1)
    # for *X, finding X$* *are,are$ onaz
    elif query[0] == '*':
        min_query = splits[1] + '$'
        max_query = splits[1] + '$' + chr(ord('z')+1)
    # for X*, finding $X*
    elif query[-1] == '*':
        min_query = '$'+ splits[0]
        max_query = '$' + splits[0][:-1] + chr(ord(splits[0][-1])+1)
    # for X*Y, finding Y$X*
    else:
        min_query = splits[1] + '$' + splits[0]
        max_query = splits[1] + '$' + splits[0][:-1] + chr(ord(splits[0][-1])+1)
    words = list(permuterm_index.items(min_query, max_query,excludemax=True))
    list_of_words = list(map(lambda x: x[1],words))

    tfidf_of_words = []
    for word in list_of_words:
        try:
            for doc in standard_inverted_index[word]:
                tfidf_of_words.append((doc,docs[row_vector_mapping[doc]][word],word))
        except:
            pass

    tfidf_of_words.sort(key = lambda x:(-x[1],x[0]))
    tfidf_of_words = tfidf_of_words[:15]
    
    end_time = time.time()

    print("\nOur Retrieval Time : ",end_time-start_time)

    print("\nTop 15 results : [Ranked on tf-idf scores]")
    for i,j,k in tfidf_of_words:
        print("-----")
        doc,row = map(int,i.split('_'))
        print("Word: ",k, "\nDocument Name : " , doc_file_mapping[doc],"\nRow Number : ",row,"\ntf-idf Score : ",j)
        df = pd.read_csv('./Dataset/'+doc_file_mapping[doc])
        print(df.iloc[row]["Snippet"])
        print("-----")

    
    print("\nOur Retrieval Time : ",end_time-start_time)



#%%

query = input("Enter your Query")
if '*' in query:
    print("Wildcard Query\n")    
    start_time = time.time()
    wildcard_query(query)
    end_time = time.time()
elif '~' == query[0]:
    print("Phrase Query\n")    
    start_time = time.time()
    ranked_phrase_query(query[1:])
    end_time = time.time()
else:
    print("Regular Query\n")    
    start_time = time.time()
    regular_query(100,query)
    end_time = time.time()

print('\nTotal time : [retrieval+metrics]',end_time-start_time)


# %%
