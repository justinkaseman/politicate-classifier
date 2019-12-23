import pandas as pd
import spacy
import re
import sys
import mlflow
import tempfile
import click
import os

from utils import printProgressBar
import en_core_web_sm

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
@click.command(help="Given a CSV file, cleans it "
                    "in an mlflow artifact called 'output-csv-dir'")
@click.option("--csv-path")
def clean_data(csv_path):
    with mlflow.start_run() as mlrun:
        df = pd.read_csv(csv_path)
        print('Cleaning ' + sys.argv[1] + " : " + sys.argv[2] + '...\n\n')

        # Instantiate spacy
        nlp = en_core_web_sm.load()

        tokensList = []

        for i, text in enumerate(df['Post Text']):
            t0 = time()
            
            # print("Processing line: " + str(i))
            # print("\tCharacters: " + str(len(text)))
            # printProgressBar(str(i), df['Post Text'].count())
            print(int(i), int(df['Post Text'].count()))
            # segment text
            doc = nlp(text)
            # print("\tSentences: " + str(len(list(doc.sents))))
            
            # remove non stop and non alpha characters
            tokens = [i.text.lower() for i in doc if i.is_alpha and not i.like_url and not i.is_stop]
            # append words of interest to tokensList
            tokensList.append(tokens)
            
            # list back to sentence strings
            for i in tokensList:
                i = ', '.join(i)
            words = set()

        d = {'Post Text': tokensList, 'Subreddit': df['Subreddit']}
        df_cleanedText = pd.DataFrame(data=d)

        df_cleanedText['Post Text'] = df_cleanedText['Post Text'].apply(', '.join)
        df_cleanedText['Post Text'] = df_cleanedText['Post Text'].str.replace(', ', ' ')

        print('Finished - persisting artifact' + '...\n\n')
        local_dir = tempfile.mkdtemp()
        local_filename = os.path.join(local_dir, 'cleaned-data.csv')
        df_cleanedText.to_csv(local_filename, sep=',', index='False')
        mlflow.log_artifacts(local_dir, "data")

        # outFile = sys.argv[2]
        # df_cleanedText.to_csv(f'{outFile}', sep=',', index='False')

if __name__ == "__main__":
    clean_data()