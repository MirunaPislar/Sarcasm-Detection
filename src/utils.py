import sys, datetime, os
import numpy as np
import classifiers
from keras.models import model_from_json
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import keras.backend as K
from collections import Counter
from pandas import read_csv
from numpy.random import seed


def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def save_file(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_data_panda(filename, shuffle=False, seed=137):
    print("Reading data from file %s..." % filename)
    data = read_csv(filename, sep="\t+", header=None, engine='python')
    data.columns = ["Set", "Label", "Text"]
    print('The shape of this data set is: ', data.shape)
    x_train, labels_train = np.array(data["Text"]), np.array(data["Label"])
    if shuffle:
        np.random.seed(seed)
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]
    return x_train, labels_train


def save_as_dataset(data, labels, filename):
    lines = []
    first_word = "TrainSet" if "train" in filename else "TestSet"
    for i in range(len(labels)):
        if data[i] is not None:
            lines.append(first_word + '\t' + str(labels[i]) + '\t' + str(data[i]))
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def save_dictionary(dictionary, filename):
    lines = []
    for k, v in dictionary.items():
        lines.append(k + '\t' + str(v))
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()


def load_dictionary(filename):
    dictionary = {}
    file = open(filename, 'r')
    lines = file.read()
    file.close()
    for line in lines.split("\n"):
        key, value = line.split("\t")
        dictionary[key] = value
    return dictionary


def save_model(model, json_name, h5_weights_name):
    model_json = model.to_json()
    with open(json_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(h5_weights_name)
    print("Saved model with json name %s, and weights %s" % (json_name, h5_weights_name))


def load_model(json_name, h5_weights_name, verbose=False):
    # In case of saved model (not to json or yaml)
    # model = models.load_model(model_path, custom_objects={'f1_score': f1_score})
    loaded_model_json = open(json_name, 'r').read()
    model = model_from_json(loaded_model_json)
    model.load_weights(h5_weights_name)
    if verbose:
        print("Loaded model with json name %s, and weights %s" % (json_name, h5_weights_name))
    return model


# Given any number of dicts, shallow copy and merge into a new dict,
# precedence goes to key value pairs in latter dicts.
# This is in case a Python3.5 version is NOT used. (needed for my access to the zCSF cluster)
def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def batch_generator(X, y, batch_size):
    # Primitive batch generator
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


# Get some idea about the max length of the train tweets
def get_max_len_info(tweets):
    print("==================================================================\n")
    sum_of_length = sum([len(l.split()) for l in tweets])
    print("Mean of train tweets: ", sum_of_length / float(len(tweets)))
    max_tweet_len = len(max(tweets, key=len).split())
    print("Max tweet length is = ", max_tweet_len)
    return max_tweet_len


def get_classes_ratio(labels):
    positive_labels = sum(labels)
    negative_labels = len(labels) - sum(labels)
    ratio = [max(positive_labels, negative_labels) / float(negative_labels),
             max(positive_labels, negative_labels) / float(positive_labels)]
    print("Class ratio: ", ratio)
    return ratio


def get_classes_ratio_as_dict(labels):
    ratio = Counter(labels)
    ratio_dict = {0: float(max(ratio[0], ratio[1]) / ratio[0]), 1: float(max(ratio[0], ratio[1]) / ratio[1])}
    print('Class ratio: ', ratio_dict)
    return ratio_dict


def extract_features_from_dict(train_features, test_features):
    # Transform the list of feature-value mappings to a vector
    vector = DictVectorizer(sparse=False)
    # Learn a list of feature name -> indices mappings and transform X_train_features
    x_train_features = vector.fit_transform(train_features).tolist()
    # Just transform the X_test_features, based on the list fitted on X_train_features
    # Disadvantage: named features not encountered during fit_transform will be silently ignored.
    x_test_features = vector.transform(test_features).tolist()
    print('Size of the feature sets: train =  ', len(x_train_features[0]), ', test = ', len(x_test_features[0]))
    return x_train_features, x_test_features


def feature_scaling(features):
    scaled_features = []
    max_per_col = []
    for i in range(len(features[0])):
        maxx = max([abs(f[i]) for f in features])
        if maxx == 0.0:
            maxx = 1.0
        max_per_col.append(maxx)
    for f in features:
        scaled_features.append([float(f[i]) / float(max_per_col[i]) for i in range(len(f))])
    return scaled_features


def run_models(train_features, train_labels, test_features, test_labels):
    class_ratio = get_classes_ratio_as_dict(train_labels)
    # class_ratio = 'balanced'
    classifiers.linear_svm(train_features, train_labels, test_features, test_labels, class_ratio)
    classifiers.logistic_regression(train_features, train_labels, test_features, test_labels, class_ratio)
    # classifiers.nonlinear_svm(train_features, train_labels, test_features, test_labels, class_ratio)


# Convert tweets into an array of indices of shape (m, max_tweet_length)
def tweets_to_indices(tweets, word_to_index, max_tweet_len):
    m = tweets.shape[0]
    tweet_indices = np.zeros((m, max_tweet_len))
    for i in range(m):
        sentence_words = [w.lower() for w in tweets[i].split()]
        j = 0
        for w in sentence_words:
            tweet_indices[i, j] = word_to_index[w]
            j = j + 1
    return tweet_indices


def encode_text_as_matrix(train_tweets, test_tweets, mode, max_num_words=None, lower=False, char_level=False):
    # Create the tokenizer
    tokenizer = Tokenizer(num_words=max_num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=lower, split=" ", char_level=char_level)
    # Fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_tweets)
    # Encode each example using a 'mode' scoring method (mode can be count, binary, freq, tf-idf)
    x_train = tokenizer.texts_to_matrix(train_tweets, mode=mode)
    x_test = tokenizer.texts_to_matrix(test_tweets, mode=mode)
    return x_train, x_test


def encode_text_as_word_indexes(train_tweets, test_tweets, max_num_words=None, lower=False, char_level=False):
    # Create the tokenizer
    tokenizer = Tokenizer(num_words=max_num_words, filters='', lower=lower, split=" ", char_level=char_level)
    # Fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_tweets)
    # Encode each example as a sequence of word indexes based on the vocabulary of the tokenizer
    x_train = tokenizer.texts_to_sequences(train_tweets)
    x_test = tokenizer.texts_to_sequences(test_tweets)
    return tokenizer, x_train, x_test


def encode_text_as_one_hot_encodings(train_tweets, test_tweets, max_num_words=None, lower=False, char_level=False):
    # Create the tokenizer
    tokenizer = Tokenizer(num_words=max_num_words, filters='', lower=lower, split=" ", char_level=char_level)
    # Fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_tweets)
    # Get the vocabulary size
    vocab_size = len(tokenizer.word_counts)
    # Encode each example as a one-hot vector
    x_train = [tokenizer.one_hot(train_example, vocab_size * 1.5) for train_example in train_tweets]
    x_test = [tokenizer.one_hot(test_example, vocab_size * 1.5) for test_example in test_tweets]
    return x_train, x_test, vocab_size


def load_glove_vectors(glove_filename='glove.6B.100d.txt'):
    # Prepare the embedding layer
    word2vec_map = {}
    path = os.getcwd()[:os.getcwd().rfind('/')]
    f = open(path + '/res/glove/' + glove_filename)
    for line in f:
        values = line.split()
        word = values[0]
        weights = np.asarray(values[1:], dtype='float32')
        word2vec_map[word] = weights
    f.close()
    print('Found %s word vectors and with embedding dimmension %s'
          % (len(word2vec_map), next(iter(word2vec_map.values())).shape[0]))
    return word2vec_map


# Compute the word-embedding matrix
def get_embeding_matrix(word2vec_map, word_to_index, embedding_dim):
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors)
    embedding_matrix = np.zeros((len(word_to_index) + 1, embedding_dim))
    for word, i in word_to_index.items():
        embedding_vector = word2vec_map.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector
        else:
            # words not found in embedding index will initialized to random values
            # from numpy.random import seed
            seed(1337603)
            embedding_matrix[i] = np.random.uniform(-1, 1, size=(1, embedding_dim))
            # print("Not found: ", word)
    return embedding_matrix


# Custom metric function
# Taken from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1_score(y_true, y_pred):
    # Recall metric. Only computes a batch-wise average of recall,
    # a metric for multi-label classification of how many relevant items are selected.
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    # Precision metric. Only computes a batch-wise average of precision,
    # a metric for multi-label classification of how many selected items are relevant.
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision*recall) / (precision+recall))


# This code allows you to see the mislabelled examples
def analyse_mislabelled_examples(x_test, y_test, y_pred):
    for i in range(len(y_test)):
        num = np.argmax(y_pred[i])
        if num != y_test[i]:
            print('Expected:', y_test[i], ' but predicted ', num)
            print(x_test[i])


def print_statistics(y, y_pred):
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average='weighted')
    recall = metrics.recall_score(y, y_pred, average='weighted')
    f_score = metrics.f1_score(y, y_pred, average='weighted')
    print('Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nF_score: %.3f\n'
          % (accuracy, precision, recall, f_score))
    print(metrics.classification_report(y, y_pred))
    return accuracy, precision, recall, f_score


def plot_training_statistics(history, plot_name, also_plot_validation=False):
    # Plot Accuracy
    plt.figure()
    plt.plot(history.history['acc'], 'k-', label='Training Accuracy')
    if also_plot_validation:
        plt.plot(history.history['val_acc'], 'r--', label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
    else:
        plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='center right')
    plt.savefig(plot_name + "_model_acc.png")

    # Plot Loss
    plt.figure()
    plt.plot(history.history['loss'], 'k-', label='Training Loss')
    if also_plot_validation:
        plt.plot(history.history['val_loss'], 'r--', label='Validation Loss')
        plt.title('Training vs Validation Loss')
    else:
        plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='center right')
    plt.savefig(plot_name + "_model_loss.png")


# This is used to plot the coefficients that have the greatest impact on a classifier like SVM
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
    plt.savefig(path + "/plots/feature_stats.png")
    # plt.show()


# Method to print the features used
# feature_options - list of True/False values; specifies which set of features is active
# feature_names - list of names for each feature set
def print_features(feature_options, feature_names):
    print("\n=========================    FEATURES    =========================\n")
    for name, value in zip(feature_names, feature_options):
        line_new = '{:>30}  {:>10}'.format(name, value)
        print(line_new)
    print("\n==================================================================\n")


# Method to print the header of the currently running model
def print_model_title(name):
    print("\n==================================================================")
    print('{:>20}'.format(name))
    print("==================================================================\n")


# Method that prints the settings for each DNN model
def print_settings(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units,
                   epochs, batch_size, dropout):
    print("==================================================================\n")
    print("Model Settings\n")
    print("==================================================================\n")
    print("Max tweet length = ", max_tweet_length)
    print("Vocab size = ", vocab_size)
    print("Embedding vector dimension = ", embedding_vector_dim)
    print("Hidden units ", hidden_units)
    print("Epochs ", epochs)
    print("Batch size ", batch_size)
    print("Dropout ", dropout)
    print("==================================================================\n")


# This allows me to print both to file and to standard output at the same time
class writer:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)

    def flush(self):
        pass


def initialize_writer(to_write_filename):
    saved = sys.stdout
    fout = open(to_write_filename, 'wt')
    sys.stdout = writer(sys.stdout, fout)
    print("Current date and time: %s\n" % str(datetime.datetime.now()))
