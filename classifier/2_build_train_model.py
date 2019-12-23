import pandas as pd
import numpy as np
import sys
import click
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json

import matplotlib.pyplot as plt
plt.style.use('ggplot')


'''
plot_history() is optional

if we want to track performance at some point

to run this just take comments off of:

history = model.fit(...) 

as well as:

plot_history(history)
'''

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='training accuracy')
    plt.plot(x, val_acc, 'r', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='training loss')
    plt.plot(x, val_loss, 'r', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

'''

Make sure to specify csv as sys.argv[1] and outdir as sys.argv[2]

sample command line usage $: python 2_build_train_model.py data/cleaned_data.csv

'''
@click.command()
@click.option("--csv-path")
def main(csv_path):
    with mlflow.start_run() as mlrun:
        # get our data into a pd dataframe
        df = pd.read_csv(csv_path)
        print('Dataframe size:', df.shape, '\nremoving null values...\n\n')
        # we have to dropna() for now because of the keras tokenizer
        # I'm unsure how to run it with keras without this step
        df = df.dropna()
        print('Dataframe size without null values shape:', df.shape, '\n\n')

        # main settings
        # for less data
        # epochs = 10
        # batch_size = 10
        # for more data
        epochs = 5
        batch_size = 64
        MAX_NB_WORDS = 50000 # The maximum number of words to be used. (most frequent)
        MAX_SEQUENCE_LENGTH = 250 # Max number of words in each post.
        EMBEDDING_DIM = 100 # This is fixed.

        # tokenize words
        # we tokenize in 1_clean_data.py
        # however currently only way I know how to get data into keras' format
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(df['Post Text'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index), '\n\n')

        # define X/Y
        X = tokenizer.texts_to_sequences(df['Post Text'].values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH) # padding adds zeros to null vectors
        print('Shape of data tensor:', X.shape)
        Y = pd.get_dummies(df['Subreddit']).values
        print('Shape of label tensor:', Y.shape, '\n\n')

        # train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
        print('splitting train/test data')
        print('training data shape:', X_train.shape,Y_train.shape)
        print('testing data shape', X_test.shape,Y_test.shape, '\n\n')

        # create model
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.5))
        model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        print(model.summary(), '\n\n')

        # print predicted (?) accuracy
        # I'm honestly confused about this number vs the number after fitting
        accr = model.evaluate(X_test,Y_test)
        print('Test set\n  Loss: {:0.3f} %\n  Accuracy: {:0.3f} %'.format((accr[0]*100),(accr[1]*100)), '\n\n')

        # train model
        print('\nFitting model\nThis may take a while...\n\n')

        '''
        FROM DOCS:
            validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate the loss and any model
            metrics on this data at the end of each epoch. 
            The validation data is selected from the last samples in the
            x and y data provided, before shuffling. 
            This argument is not supported when x is a generator or Sequence instance.
        '''
        # history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        
        # reevaluate the model
        scores = model.evaluate(X, Y)
        print('\n\nAccuracy: {:0.3f} %'.format(scores[1]*100))
        mlflow.log_metric("accuracy", scores[1]*100)

        ##################
        # saving our model
        ##################

        mlflow.keras.log_model(model, "keras-model")

        # # specify directory to save in
        # outdir = sys.argv[2]
        # # keras' native model saving tool
        # model_json = model.to_json()
        # # serialize model to JSON
        # # the keras model that's train = 'model'
        # with open(f"{outdir}model_num.json", "w") as json_file:
        #     json_file.write(model_json)
        # print("\n\nsaved model parameters to disk as JSON!\n\n")
        # # serialize weights to HDF5
        # # we can later read in weights and add them to our json model
        # model.save_weights(f'{outdir}model_num.h5')
        # print("\n\nsaved model weights to disk as HDF5!\n\n")
        
        # commenting out for now
        # I'm unclear about model.fit training vs training for history logging
        # plot accuracy during training
        # print('plotting loss/accuracy over training period \n\n')
        # plot_history(history)

if __name__ == "__main__":
    main()