import time, os, utils
from utils import run_supervised_learning_models
from dl_models import nn_bow_model
from pandas import DataFrame


path = os.getcwd()[:os.getcwd().rfind('/')]
to_write_filename = path + '/stats/bag_of_words_analysis.txt'
utils.initialize_writer(to_write_filename)

train_filename = "train.txt"
test_filename = "test.txt"
tokens_filename = "clean_original_"
data_path = path + "/res/tokens/tokens_"

# Load the data
train_tweets = utils.load_file(data_path + tokens_filename + train_filename)
test_tweets = utils.load_file(data_path + tokens_filename + test_filename)

# Make sure all words are lower-case
x_train = [t.lower() for t in train_tweets]
x_test = [t.lower() for t in test_tweets]

# Load the labels
y_train = [int(l) for l in utils.load_file(path + "/res/datasets/ghosh/labels_" + train_filename)]
y_test = [int(l) for l in utils.load_file(path + "/res/datasets/ghosh/labels_" + test_filename)]

modes = ['binary', 'count', 'tfidf', 'freq']
results = DataFrame()

# For each selection-mode, make a BoW analysis using both SVMs and a simple feed-forward NN
for mode in modes:
    utils.print_model_title("BoW Analysis for Mode %s" % mode)
    tokenizer, x_train, x_test = utils.encode_text_as_matrix(train_tweets, test_tweets, mode, lower=True)
    word_to_indices = tokenizer.word_index
    index_to_word = {i: w for w, i in word_to_indices.items()}
    start = time.time()
    run_supervised_learning_models(x_train, y_train, x_test, y_test, make_feature_analysis=True,
                                   feature_names=index_to_word, top_features=20,
                                   plot_name="/bow_models/bow_%s_" % mode)
    nn_bow_model(x_train, y_train, x_test, y_test, results, mode, save=False, plot_graph=True)
    end = time.time()
    print("BoW for %s mode completion time: %.3f s = %.3f min" % (mode, (end - start), (end - start) / 60.0))
utils.boxplot_results(results, "bow_nn_boxplot.png")
