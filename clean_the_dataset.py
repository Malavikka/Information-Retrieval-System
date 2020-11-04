#%%
import pandas as pd
import numpy as np
import string
import os 
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords') Run this line only the first time you run the script because it needs to download the stopwords. Comment for later runs.
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# %%

# Function for language pre-processing
# Function to remove stray characters, tokenizing and stemming.
def clean_the_string(sentence):
    cleaned_sentence = []
    # replacing newlines and punctuations with space
    sentence = sentence.replace('\t', ' ').replace('\n', ' ')
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, ' ')
    sentence = sentence.split()

    # removing stop words and Stemming the remaining words
    #stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    for word in sentence:
        if word not in stopwords.words('english'):# and not word.isdigit(): => Check if we need to include this condition. If we remove all digits then queries supplied by users like "What was the averaga temperature in june" can't be addressed
            #word = stemmer.stem(word)
            word = lemmatizer.lemmatize(word)
            cleaned_sentence.append(word.lower()) # Converting everything to lower case for uniformity

    cleaned_sentence = ' '.join(cleaned_sentence)
    return cleaned_sentence

# %%

# Iterating through every file in in the directory
directory = os.fsencode('./Dataset') # Path to the CSV directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if(filename.endswith(".csv")):
        print("Filename : " , filename)
        df = pd.read_csv('./Dataset/' + filename, index_col=None)
        df.drop(columns = ['URL', 'MatchDateTime', 'IAShowID', 'IAPreviewThumb'], inplace = True)
        df.head()
        # Iterating over every row of the "Snippet" column and pre-processing it.
        for i in range(len(df['Snippet'])):
            df['Snippet'][i] = clean_the_string(df['Snippet'][i])
            # print(df['Snippet'][i])
            
        # Save the modified dataframe
        df.to_csv('./Processed_dataset/' + filename, index = False)


# %%
