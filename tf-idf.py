Dict = { 'Dict1': {1: [1,2,3], 2: [4,5,6], 3: [7,8,9]}, 'Dict2': {'Name': [5,1,3], 1: [1, 2]} }

import math
for key,value in Dict.items():
    count = 0
    no_of_files = 0
    for new_key,new_value in value.items():
        count = count + len(new_value)
        no_of_files = no_of_files + 1

    tf = 1 + math.log10(count)
    idf = math.log10(417/no_of_files)
    tf_idf = tf * idf
    print(key , "TF-IDF Score : " , tf-idf)
        