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

# Add argument to allow user to input target filename.
parser.add_argument(
    "samples",
    type = int,
    default = 1000,
    help = "Number of comments to run model on.")

# Parse the arguments.
args = parser.parse_args()

# Access the values of the arguments.
samplesize = args.samples

def main():

    ##### HELPER FUNCTIONS
    # CLEAN TEXT
    # This function removes punctuation and converts every character to 
    # lowercase. Then, it converts the text into bytes using UTF-8 encoding
    # and translates it back using ASCII encoding in order to remove 
    # characters that may not be processed by certain systems or software.

    def clean_text(txt):

        txt = "".join(v for v in txt if v not in string.punctuation).lower()
        txt = txt.encode("utf8").decode(
            "ascii",
            'ignore')

        return txt 

    # GET SEQUENCE OF TOKENS
    # This function converts data to a sequence of tokens. 
    # Each token is converted into an integer. For each token, 
    # the token or sequence of tokens that come after it is 
    # found and added to the list. Essentially, you get a list
    # of possible word sequences.
    def get_sequence_of_tokens(tokenizer, corpus):
        
        input_sequences = []

        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        return input_sequences

    # GENERATE PADDED SEQUENCES
    # This function creates padding around the input sequences
    # that makes every sentence the same length. The padding
    # comes before the sequences, making the last word into the
    # target word ('label' - the word our model will generate).
    def generate_padded_sequences(input_sequences):
        
        max_sequence_len = max([
            len(x) for x in input_sequences])
        
        input_sequences = np.array(
            pad_sequences(
                input_sequences, 
                maxlen = max_sequence_len, 
                padding = 'pre'))

        predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
        label = ku.to_categorical(
            label, 
            num_classes = total_words)

        return predictors, label, max_sequence_len

    # CREATE MODEL
    # This function creates the model that will be used to
    # generate new sequences of comments. The function takes
    # following arguments:
    #  - max_sequence_len: an integer representing the maximum 
    # length of a text sequence
    # - total_words: an integer representing the total number 
    # of unique words in the vocabulary of the text corpus
    def create_model(max_sequence_len, total_words):
        
        input_len = max_sequence_len - 1
        model = Sequential() 
        
        # Add Input Embedding Layer
        # creates dense vector representation for each input 
        # word
        model.add(
            Embedding(
                total_words, # the total number of unique words 
                # in the vocabulary
                10, # the size of the vector space in which the 
                # words will be embedded
                input_length = input_len))
        
        # Add Hidden Layer 1 - LSTM Layer
        model.add(
            LSTM(
                100)) # number of memory cells in the layer
        
        # Add Dropout Layer
        # prevent overfitting by randomly dropping out some of 
        # the connections between the LSTM cells
        model.add(
            Dropout(
                0.1)) # dropout rate
        
        # Add Output Layer
        # generate a probability distribution over the 
        # vocabulary of possible next words in the sequence
        model.add(
            Dense(
                total_words, # the total number of unique 
                # words in the vocabulary
                activation = 'softmax'))

        model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam')
        
        return model

    data_dir = os.path.join("data")
    
    # FETCH DATA
    # load data
    all_comments = []
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            comment_df = pd.read_csv(
                data_dir + "/" + filename)
            all_comments.extend(
                list(
                    comment_df["commentBody"].values))

    # CLEAN DATA
    # Remove 'Unknown' comments
    all_comments = [c for c in all_comments if c != "Unknown"]
    # randomly select 100,000 points in order to
    # not crash the system
    random_comments = random.sample(
        all_comments, 
        samplesize)
    # Run helper function
    corpus = [clean_text(x) for x in random_comments]

    # PREPARE INPUT DATA
    # Tokenize corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # Get list of input sequences
    inp_sequences = get_sequence_of_tokens(
        tokenizer, 
        corpus)

    # Pad sequences, extract predictors and targets
    predictors, label, max_sequence_len = generate_padded_sequences(
        inp_sequences)

    # create model
    model = create_model(
        max_sequence_len, 
        total_words)

    # Fit model
    history = model.fit(
        predictors,
        label, 
        epochs = 100,
        batch_size = 128,
        verbose = 1)

    # save model
    model.save(
        os.path.join(
            "mdls", 
            "model.tf"))
    # save tokenizer
    dump(
        tokenizer, 
        open(
            os.path.join(
                "out", 
                "tokenizer.pickle"), 
            'wb'))
    # save max sequence length
    f = open(
        os.path.join(
            "out", 
            "max_sequence_len.txt"), 
        "w")
    f.write(str(max_sequence_len))
    f.close()

if __name__ == "__main__":
    main()
