# Assignment 3 - Language modelling and text generation using RNNs
This assignment is ***Part 3*** of the portfolio exam for ***Language Analytics S23***. The exam consists of 5 assignments in total (4 class assignments and 1 self-assigned project).

## Contribution
The initial assignment was created partially in collaboration with other students in the course, also making use of code provided as part of the course. The final code is my own. Several adjustments have been made since the initial hand-in.

Here is the link to the GitHub repository containing the code for this assignment: https://github.com/keresztalina/cds_lang_3

## Assignment description by Ross
*(NB! This description has been edited for brevity. Find the full instructions in ```README_rdkm.md```.)*

For this assignemnt, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a collection of scripts which do the following:

- Train a model on the Comments section of the data
  - [Save the trained model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- Load a saved model
  - Generate text from a user-suggested prompt

## Methods
### Model training
The purpose of this script is to intake textual data, generate sequences of words that occur after each other, and train a model to predict what word can come after another. 

First, the New York Times comments data is loaded and the text extracted into a list. Due to resource limitations, the machines available for use in this course are not always able to process the entire quantity of the available data, so a user-defined quantity of comments are randomly selected to suit the user's computational resources. The text is also cleaned. Punctuation is removed and all characters are converted to lowercase. The text  is then converted into bytes using UTF-8 encoding and translated back using ASCII encoding in order to remove characters that may not be processed by certain systems or software.

Then, the thus cleaned corpus is tokenized for more convenient representation. A list of input word sequences for the model are generated. Each token is converted into an integer. For each token, the token or sequence of tokens that come after it is found and added to the list, generating a sequence of possible words present in the corpus that can occur together in a specific order. Padding is added before these sequences in order to make each sequence the same length and to turn the last word into the target word ('label') the model is attempting to predict based on the preceding sequence.




### Text generation

## Usage
## Usage
### Prerequisites
This code was written and executed in the UCloud application's Coder Python interface (version 1.77.3, running Python version 3.9.2). UCloud provides virtual machines with a Linux-based operating system, therefore, the code has been optimized for Linux and may need adjustment for Windows and Mac.

### Installations
1. Clone this repository somewhere on your device.
2. Download the data from [here](https://www.kaggle.com/datasets/aashita/nyt-comments). Place it into a ```/data``` folder located within the repository. Your repository should have the following structure:

- ```cds_lang_3```
    - ```data```
    - ```mdls```
    - ```out```
    - ```src```

3. Open a terminal and navigate into the ```/cds_lang_3``` folder. Run the following lines in order to install the necessary packages:
        
        pip install --upgrade pip
        python3 -m pip install -r requirements.txt
        
### Run the script.
In order to run the scripts, make sure your current directory is still the ```/cds_lang_3``` folder. 

In order to train the model, from command line, run: 

        python3 src/train_model.py <NUM_COMMENTS>















