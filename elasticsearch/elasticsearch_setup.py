# import Elasticsearch module
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
from os import walk
# get all csv file names to be loaded into elasticsearch from the original datasets folder
path = "./../Dataset/"
files_list = []
for i,j,k in walk(path):
    files_list.extend(k)
# create connection to elasticsearch 
es = Elasticsearch([{'host':'localhost','port':9200}])
# for each csv we create an index and return the 
for f in files_list:
    df = pd.read_csv(path+f)
    df = df.where(pd.notnull(df), None)
    index_name = f[:-4].lower()
    documents = df.to_dict(orient='records')
    print("-------------\nIndex created: " + index_name)
    es.indices.create(index = index_name,request_timeout=30)
    print("Indexing Start: " + index_name)
    helpers.bulk(es, documents, index = index_name, doc_type='_doc', raise_on_error=True)
    print("Index finished:" + index_name +" \n-------------")