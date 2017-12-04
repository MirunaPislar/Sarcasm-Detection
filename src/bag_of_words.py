import time
import os
from pandas import DataFrame
import data_processing as data_proc
import numpy as np
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import  models
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


results = DataFrame()


def process_set(set_filename, vocabulary_filename, use_tweet_tokenize=False):
    # Read the data from the set file
    data, labels = data_proc.load_data_panda(set_filename)
    # Load the vocabulary (and build it if not there)
    vocabulary = data_proc.build_vocabulary(data, vocabulary_filename)
    print("Vocabulary (of size %d) successfully loaded from file %s." % (len(vocabulary), vocabulary_filename))
    # Process the tweets in the set
    print("Processing tweets in %s..." % set_filename)
    tweets = data_proc.process_tweets(data, vocabulary, use_tweet_tokenize)
    return tweets, labels


def plot_coefficients(classifier, feature_names, path, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=30, ha='right')
    plt.ylabel("Coefficient Value")
    plt.title("Visualising Top Features")
    plt.savefig(path + "/plots/feature_stats_sing_tweet_tknzr.png")
    plt.show()


def analyse_features(train_set, path, vocab_filename, use_tweet_tokenize):
    print("Analysing coefficients...")
    tweets, labels = process_set(train_set, vocabulary_filename=vocab_filename,
                                 use_tweet_tokenize=use_tweet_tokenize)
    cv = CountVectorizer()
    cv.fit(tweets)
    X_train = cv.transform(tweets)
    svm = LinearSVC()
    svm.fit(X_train, labels)
    plot_coefficients(svm, cv.get_feature_names(), path, top_features=5)


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


# Define f-score metric for keras model.fit
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
def nn(x_train, y_train, x_test, y_test, path, no_of_epochs = 50, batch_size = 32, mode='tfidf',
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
        plot_name = path + "/plots/plot_using_tweet_tknzr_" + mode + "_mode_" + str(no_of_epochs) + "epochs.png"
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
        json_name = path + "/models/json_bow_nn_" + mode + "_mode_" + str(no_of_epochs) + "epochs.json"
        h5_weights_name = path + "/models/h5_bow_nn_" + mode + "_mode_" + str(no_of_epochs) + "epochs.json"
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


def bag_of_words(train_set, test_set, vocab_filename, path, use_tweet_tokenize=False,
                 mode='tf-idf', use_svm=True, use_nn=True):
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
        nn(x_train, train_labels, x_test, test_labels, path, mode=mode, no_of_epochs=10,
                     batch_size=32, save=True, plot_graph=True)
    end = time.time()
    print("BoW model analysis completion time: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))


def main(use_tweet_tokenize=True, make_feature_analysis=True):
    path = os.getcwd()[:os.getcwd().rfind('/')]
    train_set = path + "/res/train.txt"
    test_set = path + "/res/test.txt"

    if use_tweet_tokenize:
        vocab_filename = path + "/res/vocabulary_tweet_tok.txt"
    else:
        vocab_filename = path + "/res/vocabulary.txt"

    modes = ['binary', 'count', 'tfidf', 'freq']
    for mode in modes:
        print("\n=======================================================\n")
        print("                      Mode :  %s" % mode)
        print("\n=======================================================\n")
        bag_of_words(train_set, test_set, vocab_filename, path, use_tweet_tokenize,
                     mode=mode, use_svm=True, use_nn=True)
    if not results.empty:
        plt.figure()
        results.boxplot()
        plt.savefig(path + "/plots/all_modes_box_plot_nn_using_tokenizer.png")
        plt.show()
    if make_feature_analysis:
        analyse_features(train_set, path, vocab_filename=vocab_filename, use_tweet_tokenize=use_tweet_tokenize)


if __name__ == "__main__":
    main()
