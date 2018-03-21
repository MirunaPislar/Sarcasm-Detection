import os, utils
import data_processing as data_proc


# Make a simple analysis of the key-features picked by models trained on embeddings (works for both emojis and words)
def rule_based_comparison(x_train, y_train, x_test, y_test, vocab_filename, verbose=False):
    # Build a vocabulary and count the sarcastic or non-sarcastic context in which a word appears
    vocab = data_proc.build_vocabulary(vocab_filename, x_train, minimum_occurrence=10)
    # vocab = set(' '.join([x.lower() for x in x_train]).split()) # this includes all words in the train set
    counts = {k: [0, 0] for k in vocab}
    for tw, y in zip(x_train, y_train):
        for word in tw.split():
            word = word.lower()
            if word in vocab:
                if y == 0:
                    counts[word][0] += 1
                else:
                    counts[word][1] += 1

    # Calculate the relative weight of each word, based on the sarcastic/non-sarcastic tweets that it appears
    weight = dict.fromkeys([k for k in counts.keys()], 0)
    for word in counts.keys():
        if counts[word][1] + counts[word][0] != 0:
            weight[word] = (counts[word][1] - counts[word][0]) / (counts[word][1] + counts[word][0])

    if verbose:
        total_sarcastic = sum([1 for y in y_train if y == 1])
        stopwords = data_proc.get_stopwords_list()
        probs = {word: (counts[word][1] / total_sarcastic) for word in counts.keys()
                 if word not in stopwords and word.isalnum()}
        print("Top 10 most sarcastic items: ", ' '.join(sorted(probs, key=probs.get, reverse=True)[:10]))

    # Rule-based predictions based on the previously calculated weigths
    y_pred = []
    for tw, y in zip(x_test, y_test):
        score = 0.0
        for word in tw.split():
            word = word.lower()
            if word in vocab:
                score += weight[word]
        if score >= 0.0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    utils.print_statistics(y_test, y_pred)


path = os.getcwd()[:os.getcwd().rfind('/')]
to_write_filename = path + '/stats/key_features_analysis_rule_based.txt'
utils.initialize_writer(to_write_filename)

train_filename = "train.txt"
test_filename = "test.txt"
tokens_filename = "clean_original_"
data_path = path + "/res/tokens/tokens_"
vocab_filename = path + "/res/vocabulary/vocabulary.txt"

# Load the data
train_tweets = utils.load_file(data_path + tokens_filename + train_filename)
test_tweets = utils.load_file(data_path + tokens_filename + test_filename)

# Load the labels
train_labels = [int(l) for l in utils.load_file(path + "/res/datasets/ghosh/labels_" + train_filename)]
test_labels = [int(l) for l in utils.load_file(path + "/res/datasets/ghosh/labels_" + test_filename)]

# A rule-based approach used here to analyse the key-features that are actually learnt in a (non-)sarcastic context
utils.print_model_title("Rule-based analysis")
rule_based_comparison(train_tweets, train_labels, test_tweets, test_labels, vocab_filename)
