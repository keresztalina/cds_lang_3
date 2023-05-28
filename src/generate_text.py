##### LOAD MODULES
# data processing tools
import string, os 
import pandas as pd
import numpy as np
import random
from joblib import dump
import pickle
import argparse
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# supress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(
    action = 'ignore', 
    category = FutureWarning)

##### ARGUMENT PARSER
# Create ArgumentParser object.
parser = argparse.ArgumentParser()

# Add argument to allow user to input starter word.
parser.add_argument(
    "word",
    type = str,
    default = "Denmark",
    help = "Starter word for text generation.")

# Add argument to allow user to input starter word.
parser.add_argument(
    "number_next_words",
    type = int,
    default = 5,
    help = "Number of words to generate after starter word.")

# Parse the arguments.
args = parser.parse_args()

# Access the values of the arguments.
seedtext = args.word
nextwords = args.number_next_words

##### MAIN
def main():

    # GENERATE TEXT
    # This function uses the previously defined model to 
    # generate a new piece of text. It takes the following 
    # arguments: 
    # - seed_text: a string representing the starting text 
    # for text generation
    # - next_words: an integer representing the number of 
    # words to generate after the seed text
    # - model: a trained Keras neural network model that 
    # will be used for text generation
    # - max_sequence_len: an integer representing the maximum 
    # length of the input sequence that the model was trained 
    # on

    def generate_text(seed_text, next_words, model, max_sequence_len):
        for _ in range(next_words):
            
            token_list = tokenizer.texts_to_sequences([
                seed_text])[0]
            
            # pad the sequence with zeros to match the length 
            # of the input sequences that the model was 
            # trained on
            token_list = pad_sequences([
                token_list],
                maxlen = max_sequence_len-1,
                padding = 'pre') 

            # determine the index of the word with the highest 
            # predicted probability in the output vocabulary
            predicted = np.argmax(
                model.predict(
                    token_list),
                    axis = 1)
            
            # look up the actual word corresponding to the 
            # predicted index in the tokenizer's word index, 
            # then append this word to the seed_text variable
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word

        return seed_text.title()

    ##### LOAD DATA
    # Load model
    model = tf.keras.models.load_model(
        os.path.join(
            "mdls", 
            "model.tf"))

    # Load tokenizer
    with open(
        os.path.join(
            "out", 
            "tokenizer.pickle"), 'rb') as t:
        tokenizer = pickle.load(t)

    # Load max sequence length
    with open(
        os.path.join(
            "out",
            "max_sequence_len.txt")) as f:
        max_sequence_len = f.read()
    # convert to integer    
    max_sequence_len = int(max_sequence_len)

    ##### RUN MODEL
    print(
        generate_text(
            seedtext, 
            nextwords, 
            model, 
            max_sequence_len))

if __name__ == "__main__":
    main()