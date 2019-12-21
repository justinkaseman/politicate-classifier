import pandas as pd
import numpy as np
import sys
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential

# in order to process input text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

'''

sample command line usage $: python 3_predictor.py output/ "hello my name turd ferguson and i still use yahoo."

next step will be to add file reading capabilities starting with .txt files.

Then make it able to read multiple file types.

For reaeding in the new text input I'm currently only running it through Keras' tokenizer and not really doing any other preprocessing.

Will add preprocessing functionality later.

'''
def main():
    MAX_NB_WORDS = 50000 # The maximum number of words to be used. (most frequent)
    MAX_SEQUENCE_LENGTH = 250 # Max number of words in each post.
    # # dataframe
    # csvPath = sys.argv[1]
    # print('\nreading ' + csvPath + '...\n\n')
    # df = pd.read_csv(csvPath)
    # print('Dataframe size:', df.shape, '\nremoving null values...\n\n')
    # df = df.dropna()
    # print('Dataframe size without null values shape:', df.shape, '\n\n')

    # loading model
    modelPath = sys.argv[1]
    
    # load json and create model
    json_file = open(f'{modelPath}model_num.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into the new model
    loaded_model.load_weights(f'{modelPath}model_num.h5')
    print("\n\nLoaded model from disk!\n\n")

    # evaluate the new model with random data
    xTrain = np.random.rand(200,10)
    yTrain = np.random.rand(200,1)
    xVal = np.random.rand(100,250)
    yVal = np.random.rand(100,2)

    # xTrain = xTrain.reshape(len(xTrain), 1, xTrain.shape[1])
    # xVal = xVal.reshape(len(xVal), 1, xVal.shape[1])

    # yFit = loaded_model.predict(xVal, batch_size=10, verbose=1)
    # print()
    # print(yFit)
    loaded_model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    print('EVALUATING NEW MODEL...')            
    accr = loaded_model.evaluate(xVal,yVal)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    print('\n\n', loaded_model.summary())
        

    
    # predict new data
    # inputPredictionText = ['What about the next abusive corporation/executives and not just fossil fuel? We need to play the long game and quit this cat and mouse game.']
    inputPredictionText = sys.argv[2]

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    seq = tokenizer.texts_to_sequences(inputPredictionText)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = loaded_model.predict(padded)
    labels = ['republican', 'democrat']
    # set numpy to use 3 decimals for output
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(f'\ninput: {inputPredictionText}', '\n')
    print(f'prediction: {labels[np.argmax(pred)]}', f'\nmodel.predict(padded): {pred[0]}\n\n')

if __name__ == "__main__":
    main()