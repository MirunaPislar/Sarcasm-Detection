import sys, datetime, os, math
import numpy as np
import classifiers
import data_processing as data_proc
from keras.models import model_from_json
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import keras.backend as K
from collections import Counter, OrderedDict
from pandas import read_csv
from numpy.random import seed
path = os.getcwd()[:os.getcwd().rfind('/')]


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
    seed(1655483)
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


def run_supervised_learning_models(train_features, train_labels, test_features, test_labels,
                                   make_feature_analysis=False, feature_names=None, top_features=0, plot_name="coeff"):
    class_ratio = get_classes_ratio_as_dict(train_labels)   # alternatively, can be set class_ratio = 'balanced'
    classifiers.linear_svm(train_features, train_labels, test_features, test_labels, class_ratio,
                           make_feature_analysis, feature_names, top_features, plot_name)
    classifiers.logistic_regression(train_features, train_labels, test_features, test_labels, class_ratio,
                                    make_feature_analysis, feature_names, top_features, plot_name)
    # classifiers.nonlinear_svm(train_features, train_labels, test_features, test_labels, class_ratio,
    #                          make_feature_analysis, feature_names, top_features, plot_name)


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
    return tokenizer, x_train, x_test


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


# Build random vector mappings of a vocabulary
def build_random_word2vec(tweets, embedding_dim=100, variance=1):
    print("\nBuilding random vector of mappings with dimension %d..." % embedding_dim)
    word2vec_map = {}
    seed(1457873)
    words = set((' '.join(tweets)).split())
    for word in words:
        embedding_vector = word2vec_map.get(word)
        if embedding_vector is None:
            word2vec_map[word] = np.random.uniform(-variance, variance, size=(embedding_dim,))
    return word2vec_map


def load_vectors(filename='glove.6B.100d.txt'):
    print("\nLoading vector mappings from %s..." % filename)
    word2vec_map = {}
    if 'glove' in filename:
        f = open(path + '/res/glove/' + filename)
    elif 'emoji' in filename:
        f = open(path + '/models/emoji2vec/' + filename)
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
def get_embedding_matrix(word2vec_map, word_to_index, embedding_dim, init_unk=True, variance=None):
    # Get the variance of the embedding map
    if init_unk and variance is None:
        variance = embedding_variance(word2vec_map)
        print("Word vectors have variance ", variance)
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors)
    embedding_matrix = np.zeros((len(word_to_index) + 1, embedding_dim))
    for word, i in word_to_index.items():
        embedding_vector = word2vec_map.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        elif init_unk:
            # Unknown tokens are initialized randomly by sampling from a uniform distribution [-var, var]
            seed(1337603)
            embedding_matrix[i] = np.random.uniform(-variance, variance, size=(1, embedding_dim))
        # else:
        #    print("Not found: ", word)
    return embedding_matrix


# Get the vec representation of a set of tweets based on a specified embedding (can be a word or emoji mapping)
def get_tweets_embeddings(tweets, vec_map, embedding_dim=100, init_unk=False, variance=None, weighted_average=True):
    # Get the variance of the embedding map
    if init_unk and variance is None:
        variance = embedding_variance(vec_map)
        print("Vector mappings have variance ", variance)
    # If set, calculate the tf-idf weight of each embedding, otherwise, no weighting (all weights are 1.0)
    if weighted_average:
        weights = get_tf_idf_weights(tweets, vec_map)
    else:
        weights = {k: 1.0 for k in vec_map.keys()}
    tw_emb = np.zeros((len(tweets), embedding_dim))
    for i, tw in enumerate(tweets):
        total_valid = 0
        for word in tw.split():
            embedding_vector = vec_map.get(word)
            if embedding_vector is not None:
                tw_emb[i] = tw_emb[i] + embedding_vector * weights[word]
                total_valid += 1
            elif init_unk:
                seed(1337603)
                tw_emb[i] = np.random.uniform(-variance, variance, size=(1, embedding_dim))
            # else:
            #    print("Not found: ", word)
        # Get the average embedding representation for this tweet
        tw_emb[i] /= float(max(total_valid, 1))
    return tw_emb


# Based on the deepmoji project, predicting emojis for each tweet -- done using their pre-trained weights
# Here we extract the relevant emojis (with an individual probability of being accurate over teh set threshold)
def get_deepmojis(filename, threshold=0.05):
    print("\nGetting deep-mojis for each tweet in %s..." % filename)
    df = read_csv(path + "/res/deepmoji/" + filename, sep='\t')
    pred_mappings = load_file(path + "/res/emoji/wanted_emojis.txt").split("\n")
    emoji_pred = []
    for index, row in df.iterrows():
        tw_pred = []
        for top in range(5):
            if row['Pct_%d' % (top+1)] >= threshold:
                tw_pred.append(row['Emoji_%d' % (top + 1)])
        emoji_pred.append([pred_mappings[t] for t in tw_pred])
    print("Couldn't find a strong emoji prediction for %d emojis" % len([pred for pred in emoji_pred if pred == []]))
    return emoji_pred


# Calculate the variance of an embedding (like glove, word2vec, emoji2vec, etc)
# Used to sample new uniform distributions of vectors in the interval [-variance, variance]
def embedding_variance(vec_map):
    variance = np.sum([np.var(vec) for vec in vec_map.values()]) / len(vec_map)
    return variance


# Shuffle the words in all tweets
def shuffle_words(tweets):
    shuffled = []
    for tweet in tweets:
        words = [word for word in tweet.split()]
        np.random.shuffle(words)
        shuffled.append(' '.join(words))
    return shuffled


# Get the tf-idf weighting scheme (used to measure the contribution of a word in a tweet => weighted sum of embeddings)
def get_tf_idf_weights(tweets, vec_map):
    df = {}
    for tw in tweets:
        words = set(tw.split())
        for word in words:
            if word not in df:
                df[word] = 0.0
            df[word] += 1.0
    idf = OrderedDict()
    for word in vec_map.keys():
        n = 1.0
        if word in df:
            n += df[word]
        score = math.log(len(tweets) / float(n))
        idf[word] = score
    return idf


# Compute the similarity of 2 vectors, both of shape (n, )
def cosine_similarity(u, v):
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    cosine_distance = dot / (norm_u * norm_v)
    return cosine_distance


# Performs the word analogy task: a is to b as c is to ____.
def complete_analogy(a, b, c, vec_map):
    # Get the vector embeddings
    e_a, e_b, e_c = vec_map[a], vec_map[b], vec_map[c]
    max_cosine_sim = -100
    best = None
    for v in vec_map.keys():
        # The best match shouldn't be one of the inputs, so pass on them.
        if v in [a, b, c]:
            continue
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)
        cosine_sim = cosine_similarity(e_b - e_a, vec_map[v] - e_c)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best = v
    print(str.format('{} - {} + {} = {}', a, b, c, best))
    return best


# Get the Euclidean distance between two vectors
def euclidean_distance(u_vector, v_vector):
    distance = np.sqrt(np.sum([(u - v) ** 2 for u, v in zip(u_vector, v_vector)]))
    return distance


# Given a tweet, return the scores of the most similar/dissimilar pairs of words
def get_similarity_measures(tweet, vec_map, weighted=False, verbose=True):
    # Filter a bit the tweet so that no punctuation and no stopwords are included
    stopwords = data_proc.get_stopwords_list()
    filtered_tweet = list(set([w.lower() for w in tweet.split()
                      if w.isalpha() and w not in stopwords and w.lower() in vec_map.keys()]))
    # Compute similarity scores between any 2 words in filtered tweet
    similarity_scores = []
    max_score = -100
    min_score = 100
    for i in range(len(filtered_tweet) - 1):
        wi = filtered_tweet[i]
        for j in range(i + 1, len(filtered_tweet)):
            wj = filtered_tweet[j]
            similarity = cosine_similarity(vec_map[wi], vec_map[wj])
            if weighted:
                similarity /= euclidean_distance(vec_map[wi], vec_map[wj])
            similarity_scores.append(similarity)
            if max_score < similarity:
                max_score = similarity
                max_words = [wi, wj]
            if min_score > similarity:
                min_score = similarity
                min_words = [wi, wj]
    if verbose:
        print("Filtered tweet: ", filtered_tweet)
        if max_score != -100:
            print("Maximum similarity is ", max_score, " between words ", max_words)
        else:
            print("No max! Scores are: ", similarity_scores)
        if min_score != 100:
            print("Minimum similarity is ", min_score, " between words ", min_words)
        else:
            print("No min! Scores are: ", similarity_scores)
    return max_score, min_score


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


def plot_training_statistics(history, plot_name, also_plot_validation=False, acc_mode='acc', loss_mode='loss'):
    # Plot Accuracy
    plt.figure()
    plt.plot(history.history[acc_mode], 'k-', label='Training Accuracy')
    if also_plot_validation:
        plt.plot(history.history['val_' + acc_mode], 'r--', label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
    else:
        plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='center right')
    plt.ylim([0.0, 1.0])
    plt.savefig(path + plot_name + "_acc.png")

    # Plot Loss
    plt.figure()
    plt.plot(history.history[loss_mode], 'k-', label='Training Loss')
    if also_plot_validation:
        plt.plot(history.history['val_' + loss_mode], 'r--', label='Validation Loss')
        plt.title('Training vs Validation Loss')
    else:
        plt.title('Training Loss ')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='center right')
    plt.savefig(path + plot_name + "_loss.png")


# This is used to plot the coefficients that have the greatest impact on a classifier like SVM
# feature_names = a dictionary of indices/feature representations to words (or whatever you're extracting features from)
def plot_coefficients(classifier, feature_names, top_features=20, plot_name="/bow_models/bow_binary_", show=False):
    # Get the top most positive/negative coefficients
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    x_names = [feature_names[feature] for feature in top_coefficients]

    # Plot the coefficients
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    plt.xticks(np.arange(0, 2 * top_features), x_names, rotation=30, ha='right')
    plt.ylabel("Coefficient Value")
    plt.title("Visualising the top %d features taken up by an SVM model" % top_features)
    plt.savefig(path + "/plots/" + plot_name + "top%d_coefficients.png" % top_features)
    if show:
        plt.show()


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
def print_settings(max_tweet_length, embedding_vector_dim, hidden_units,
                   epochs, batch_size, dropout, emb_type, trainable):
    print_model_title("Settings")
    print("Max tweet length = ", max_tweet_length)
    print("Embedding vector dimension = ", embedding_vector_dim)
    print("Hidden units = ", hidden_units)
    print("Epochs = ", epochs)
    print("Batch size = ", batch_size)
    print("Dropout = ", dropout)
    print("Embeddings type = ", emb_type)
    print("Trainable = ", trainable)
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
