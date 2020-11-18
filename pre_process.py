#%%
# Pre-processing steps being followed :
# 1) Converting everything to lower case
# 2) Removed numbers
# 3) Removed punctuation
# 4) Removed whitespaces
# 5) Removed Stopwords
# 6) Did porter Stemming

#%%
import pandas as pd
import os
import nltk 
import string 
import re 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

#%%
def clean_the_string(sentence):
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

    cleaned_sentence = ' '.join(cleaned_sentence)
    return cleaned_sentence
    
#%%
directory = os.fsencode('./Dataset') # Path to the CSV directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if(filename.endswith(".csv")):
        print("Filename : " , filename)
        df = pd.read_csv('./Dataset/' + filename, index_col=None)
        df.drop(columns = ['URL', 'MatchDateTime', 'IAShowID', 'IAPreviewThumb'], inplace = True)
        # Iterating over every row of the "Snippet" column and pre-processing it.
        for i in range(len(df['Snippet'])):
            df['Snippet'][i] = clean_the_string(df['Snippet'][i])
            # print(df['Snippet'][i])
            
        # Save the modified dataframe
        df.to_csv('./New_Processed_data/' + filename, index = False)

# %%
