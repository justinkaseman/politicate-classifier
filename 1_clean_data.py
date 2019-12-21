import pandas as pd
import spacy
import re
import sys

from gensim.sklearn_api import W2VTransformer
from time import time

# nltk stuff
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))

'''

The data input needs to be a csv at this point.

.csv file needs to include column called "Post Text" and another column called "Subreddit"

code will iterate through ['Post Text'] series and clean that text. Subreddit will be used for classification in 2_build_train_model.py

sample command line usage $: python 1_build_train_model.py output/path/cleaned_file.csv

'''
def main():
    csvPath = sys.argv[1]
    print('\nreading' + csvPath + '...\n\n')
    df = pd.read_csv(csvPath)
    print('Cleaning ' + sys.argv[1] + '...\n\n')

    # Instantiate spacy
    nlp = spacy.load("en_core_web_sm")

    tokensList = []

    for i, text in enumerate(df['Post Text']):
        t0 = time()
        
        print("Processing line: " + str(i))
        print("\tCharacters: " + str(len(text)))

        # segment text
        doc = nlp(text)
        print("\tSentences: " + str(len(list(doc.sents))))
        
        # remove non stop and non alpha characters
        tokens = [i.text.lower() for i in doc if i.is_alpha and not i.like_url and not i.is_stop]
        # append words of interest to tokensList
        tokensList.append(tokens)
        
        # list back to sentence strings
        for i in tokensList:
            i = ', '.join(i)
        words = set()
        print("\tWords: " + str(len(words)))
        print("\tProcess Time:" + str(time() - t0) + "\n")

    d = {'Post Text': tokensList, 'Subreddit': df['Subreddit']}
    df_cleanedText = pd.DataFrame(data=d)

    df_cleanedText['Post Text'] = df_cleanedText['Post Text'].apply(', '.join)
    df_cleanedText['Post Text'] = df_cleanedText['Post Text'].str.replace(', ', ' ')

    outFile = sys.argv[2]
    df_cleanedText.to_csv(f'{outFile}', sep=',', index='False')

if __name__ == "__main__":
    main()