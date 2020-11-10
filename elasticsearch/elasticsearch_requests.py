from elasticsearch import Elasticsearch
es = Elasticsearch([{'host':'localhost','port':9200}])
body = {"query": { "bool": { "must": [{ "match": { "Snippet": "globl warm"}}]}}}
x = es.search(body=None, index='')
a = open("Results.txt",'w')
for i in x['hits']['hits']:
    print(i,file=a)
print(file=a)
print(x,file=a)
a.close()
