import time
import os
import re
from pandas import read_csv
from pandas import DataFrame
import numpy as np
import keras.backend as K
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import  models
from nltk.tokenize import TweetTokenizer
from sklearn import metrics
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


results = DataFrame()


def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_data_panda(filename):
    print("Reading data from file %s..." % filename)
    data = read_csv(filename, sep="\t", header=None)
    data.columns = ["Set", "Label", "Text"]
    print('The shape of the train set is: ', data.shape)
    print('Columns: ', data.columns.values)
    return data


def clean_tweet(tweet, clean_hashtag=False, clean_mentions=False, lower_case=False):
    # Add white space before every punctuation sign so that we can split around it and keep it
    tweet = re.sub('([!?*&%"~`^+{}])', r' \1 ', tweet)
    tweet = re.sub('\s{2,}', ' ', tweet)
    tokens = tweet.split()
    valid_tokens = []
    for word in tokens:
        if word.startswith('#') and clean_hashtag:      # do not include any hash tags
            continue
        if word.lower().startswith('#sarca'):           # do not include any #sarca* hashtags
            continue
        if clean_mentions and word.startswith('@'):     # replace all mentions with a general user
            word = '@user'
        if word.startswith('http'):                     # skip URLs
            continue

        # Process each word so it does not contain any kind of punctuation or inappropriately merged symbols
        split_non_alnum = []
        if (not word[0].isalnum()) and word[0] != '#' and word[0] != '@':
            index = 0
            while index < len(word) and not word[index].isalnum():
                split_non_alnum.append(word[index])
                index = index + 1
            word = word[index:]
        if len(word) > 1 and not word[len(word) - 1].isalnum():
            index = len(word) - 1
            while index >= 0 and not word[index].isalnum():
                split_non_alnum.append(word[index])
                index = index - 1
            word = word[:(index + 1)]
        if word != '':
            if lower_case:
                valid_tokens.append(word.lower())
            else:
                valid_tokens.append(word)
        if split_non_alnum != []:
            valid_tokens.extend(split_non_alnum)
    return valid_tokens


def build_vocabulary(data, vocab_filename, use_tweet_tokenize=False):
    print("Building vocabulary...")
    vocabulary = Counter()
    if use_tweet_tokenize:
        tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    for tweet in data:
        if use_tweet_tokenize:
            clean_tw = tknzr.tokenize(tweet)
            clean_tw = [tw for tw in clean_tw if not tw.startswith('#sarca') and not tw.startswith('http')]
        else:
            clean_tw = clean_tweet(tweet)
        vocabulary.update(clean_tw)
    save_vocab(vocabulary.keys(), vocab_filename)
    print("Vocabulary saved to file \"%s\"" %vocab_filename)
    print("The top 50 most common words: ", vocabulary.most_common(50))


def save_vocab(lines, filename):
    # Convert lines to a single blob of text
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def process_tweets(data, vocabulary, use_tweet_tokenize=False):
    tweets = list()
    if use_tweet_tokenize:
        tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    for tweet in data:
        if use_tweet_tokenize:
            clean_tw = tknzr.tokenize(tweet)
            clean_tw = [tw for tw in clean_tw if not tw.startswith('#sarca') and not tw.startswith('http')]
        else:
            clean_tw = clean_tweet(tweet)
        tokens = [word for word in clean_tw if word in vocabulary]
        tweets.append(' '.join(tokens))
    return tweets


def process_set(set_filename, vocabulary_filename, use_tweet_tokenize=False):
    # Read the data from the set file
    data = load_data_panda(set_filename)

    # Build and load the vocabulary
    if not os.path.exists(vocabulary_filename):
        build_vocabulary(data["Text"], vocabulary_filename)
    vocabulary = load_file(vocabulary_filename)
    vocabulary = set(vocabulary.split())
    print("Vocabulary (of size %d) successfully loaded from file %s." % (len(vocabulary), vocabulary_filename))

    # Process the tweets in the set
    print("Processing tweets in %s..." % set_filename)
    tweets = process_tweets(data["Text"], vocabulary, use_tweet_tokenize)

    # Get the labels for each tweet
    labels = np.array(data["Label"])

    return tweets, labels


def print_statistics(y, y_pred, printOnScreen = True):
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average='weighted')
    recall = metrics.recall_score(y, y_pred, average='weighted')
    f_score = metrics.f1_score(y, y_pred, average='weighted')
    if printOnScreen:
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F_score:', f_score)
        print(metrics.classification_report(y, y_pred))
    return accuracy, precision, recall, f_score


# Fit and evaluate SVC model
def svm(Xtrain, Ytrain, Xtest, Ytest):
    print("\n=== Linear SVC MODEL === ")
    eval_start_time = time.time()
    linear_svc = LinearSVC()
    print("Fitting Linear SVC...")
    linear_svc.fit(Xtrain, Ytrain)
    print("SVC evaluation...")
    Yhat = linear_svc.predict(Xtest)
    print_statistics(Ytest, Yhat)
    eval_end_time = time.time()
    print("BoW with SVC completion time: %.3f s = %.3f min"
          % ((eval_end_time - eval_start_time),(eval_end_time - eval_start_time) / 60.0))


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def plot_training_statistics(history, plot_name):
    plt.figure()
    plt.plot(history.history['acc'], 'k-', label='Training Accuracy')
    plt.plot(history.history['loss'], 'r--', label='Training Loss')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc='center right')
    plt.savefig(plot_name)
    plt.show()


# Fit and evaluate feed-forward Neural Network model
def nn(x_train, y_train, x_test, y_test, path, no_of_epochs = 50, batch_size = 32,
       hl_activation_function='relu', ol_activation_function='sigmoid', save=True, plot_graph=True):
    print("\n=== Feed-forward NN model ===")
    print("List of architectural choices for this run: ")
    print("no. of epochs = %d, batch size = %d, hidden layer activation = %s, output layer activation = %s."
          % (no_of_epochs, batch_size, hl_activation_function, ol_activation_function))
    eval_start_time = time.time()
    # Build the model
    model = Sequential()
    # Use a single hidden layer with 50 neurons
    model.add(Dense(50, input_shape=(x_train.shape[1],), activation=hl_activation_function))
    # The output layer is a single neuron
    model.add(Dense(1, activation=ol_activation_function))
    model.summary()
    # Train using: binary cross entropy loss, Adam implementation of Gradient Descent
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
    # Fit the model on the training data
    print("Fitting feed-forward NN model...")
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=no_of_epochs, verbose=2)

    if plot_graph:
        plot_name = path + "/plots/plot1_" + mode + "_" + str(no_of_epochs) + "epochs_basic.png"
        plot_training_statistics(history, plot_name)

    # Evaluate the model so far
    print("NN evaluation...")
    loss, acc, f1 = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    results[mode] = [loss, acc, f1]
    classes = model.predict_classes(x_test, batch_size=batch_size)
    y_pred = [item for c in classes for item in c]
    print_statistics(y_test, y_pred)
    print("No of examples predicted correctly: ", np.sum(y_test == y_pred))
    print("Accuracy, mine = %.3f, keras = %.3f." % (np.sum(y_test == y_pred) * 100.0 / y_test.size, acc * 100.0))

    if save:
        json_name = path + "/models/json_bow_nn_2_" + mode + "_" + str(no_of_epochs) + "_basic_18nov.json"
        h5_weights_name = path + "/models/h5_bow_2_nn_" + mode + "_" + str(no_of_epochs) + "_basic_18nov.json"
        save_model(model, json_name=json_name, h5_weights_name=h5_weights_name)

    eval_end_time = time.time()
    print("Bow with NN completion time: %.3f s = %.3f min"
          % ((eval_end_time - eval_start_time), (eval_end_time - eval_start_time) / 60.0))


def save_model(model, json_name, h5_weights_name):
    model_json = model.to_json()
    with open(json_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(h5_weights_name)
    print("Saved model with json name %s, and weights %s" % (json_name, h5_weights_name))


def load_model(json_name, h5_weights_name):
    # In case of saved model (not to json or yaml)
    # model = models.load_model(model_path, custom_objects={'f1_score': f1_score})
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(h5_weights_name)
    print("Loaded model with json name %s, and weights %s" % (json_name, h5_weights_name))
    return model


def bag_of_words(train_set, test_set, vocab_filename, path, use_tweet_tokenize=False, mode='tf-idf', use_svm=True, use_nn=True):
    start = time.time()

    # Process and prepare the training and testing sets
    train_tweets, train_labels = process_set(train_set, vocabulary_filename=vocab_filename, use_tweet_tokenize=use_tweet_tokenize)
    test_tweets, test_labels = process_set(test_set, vocabulary_filename=vocab_filename, use_tweet_tokenize=use_tweet_tokenize)

    # Create the tokenizer
    tokenizer = Tokenizer()
    # Fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_tweets)
    # Encode training data - score words based on 'mode' scoring method
    x_train = tokenizer.texts_to_matrix(train_tweets, mode=mode)
    x_test = tokenizer.texts_to_matrix(test_tweets, mode=mode)

    if use_svm:
        svm(x_train, train_labels, x_test, test_labels)
    if use_nn:
        nn(x_train, train_labels, x_test, test_labels, path, no_of_epochs=10,
                     batch_size=32, save=True, plot_graph=True)

    end = time.time()
    print("BoW model analysis completion time: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))


path = os.getcwd()[:os.getcwd().rfind('/')]
train_set = path + "/res/train.txt"
test_set = path + "/res/test.txt"
use_tweet_tokenize = False
if use_tweet_tokenize:
    vocab_filename = path + "/res/vocabulary_tweet_tok.txt"
else:
    vocab_filename = path + "/res/vocabulary.txt"

modes = ['binary', 'count', 'tfidf', 'freq']
for mode in modes:
    print("\n=======================================================\n")
    print("                      Mode :  %s" % mode)
    print("\n=======================================================\n")
    bag_of_words(train_set, test_set, vocab_filename, path, use_tweet_tokenize, mode=mode, use_svm=True, use_nn=True)
if not results.empty:
    plt.figure()
    results.boxplot()
    plt.savefig(path + "/plots/all_modes_box_plot_nn_10epochs.png")
    plt.show()
