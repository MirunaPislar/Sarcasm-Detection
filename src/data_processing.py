import re, os
import numpy as np
from pandas import read_csv
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
import split_hashtags as splitter


strong_affirmatives = ["yes", "yeah", "always", "all", "any", "every", "everybody", "everywhere", "ever"]

strong_negations = ["no", "not", "never", "none" "n't", "nothing", "neither", "nobody", "nowhere"]

punctuation = ["?", "!", "..."]

interjections = ["oh", "hey", "wow", "aha", "aham", "aw", "bam", "blah", "bingo", "boo", "bravo",
                 "cheers", "congratulations", "congrats", "duh", "eh", "gee", "gosh", "hey", "hmm",
                 "huh", "hurray", "oh", "oh dear", "oh my", "oh well", "oops", "ouch", "ow", "phew",
                 "shh", "uh", "uh-huh", "mhm", "ugh", "well", "wow", "woah", "yeah", "yep", "yikes", "yo"]

intensifiers = ["amazingly", "astoundingly", "awful", "bare", "bloody", "crazy", "dreadfully",
                "colossally", "especially", "exceptionally", "excessively", "extremely",
                "extraordinarily", "fantastically", "frightfully", "fucking", "fully", "hella",
                "holy", "incredibly", "insanely", "literally", "mightily", "moderately", "most",
                "outrageously", "phenomenally", "precious", "quite", "radically", "rather",
                "really", "remarkably", "right", "sick", "strikingly", "super", "supremely",
                "surprisingly", "terribly", "terrifically", "too", "totally", "uncommonly",
                "unusually", "veritable", "very", "wicked"]

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


def load_data_panda(filename):
    print("Reading data from file %s..." % filename)
    data = read_csv(filename, sep="\t+", header=None, engine='python')
    data.columns = ["Set", "Label", "Text"]
    print('The shape of this data set is: ', data.shape)
    return np.array(data["Text"]), np.array(data["Label"])


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


def build_subj_dicionary(lines):
    subj_dict = dict()
    lines = lines.split("\n")
    for line in lines:
        splits = line.split(' ')
        if len(splits) == 6:
            word = splits[2][6:]        # the word analyzed
            word_type = splits[0][5:]   # weak or strong subjective
            pos = splits[3][5:]         # part of speech: noun, verb, adj, adv or anypos
            polarity = splits[5][14:]   # its polarity: can be positive, negative or neutral
            new_dict_entry = {pos: [word_type, polarity]}
            if word in subj_dict.keys():
                subj_dict[word].update(new_dict_entry)
            else:
                subj_dict[word] = new_dict_entry
    return subj_dict


def get_subj_lexicon():
    filename = path + "/res/subjectivity_lexicon.tff"
    lexicon = load_file(filename)
    subj_dict = build_subj_dicionary(lexicon)
    return subj_dict


def get_emoji_dictionary():
    filename = path + "/res/emoji_list.txt"
    emojis = load_file(filename).split("\n")
    emoji_dict = {}
    for line in emojis:
        line = line.split(" ", 1)
        emoji = line[0]
        description = line[1]
        emoji_dict[emoji] = description
    return emoji_dict


def build_emoji_sentiment_dictionary():
    new_emoji_sentiment_filename = path + "/res/emoji_sentiment_dictionary.txt"
    if not os.path.exists(new_emoji_sentiment_filename):
        filename = path + "/res/emoji_sentiment_raw.txt"
        emojis = load_file(filename).split("\n")[1:]
        lines = []
        for line in emojis:
            line = line.split(",")
            emoji = line[0]
            occurences = line[2]
            negative = float(line[4]) / float(occurences)
            neutral = float(line[5]) / float(occurences)
            positive = float(line[6]) / float(occurences)
            description = line[7]
            lines.append(str(emoji) + "\t" + str(negative) + "\t" + str(neutral)
                         + "\t" + str(positive) + "\t" + description.lower())
        save_file(lines, new_emoji_sentiment_filename)
    emoji_sentiment_data = load_file(new_emoji_sentiment_filename).split("\n")
    emoji_sentiment_dict = {}
    for line in emoji_sentiment_data:
        line = line.split("\t")
        # Get emoji characteristics as a list [negative, neutral, positive, description]
        emoji_sentiment_dict[line[0]] = [line[1], line[2], line[3], line[4]]
    return emoji_sentiment_dict


def build_vocabulary(vocab_filename, lines, minimum_occurrence=1):
    if not os.path.exists(vocab_filename):
        print("Building vocabulary...")
        vocabulary = Counter()
        for line in lines:
            vocabulary.update(line)
        print("The top 10 most common words: ", vocabulary.most_common(10))
        # Filter all words that appear too rarely to be conclusive
        vocabulary = {key: vocabulary[key] for key in vocabulary if vocabulary[key] >= minimum_occurrence}
        save_file(vocabulary.keys(), vocab_filename)
        print("Vocabulary saved to file \"%s\"" % vocab_filename)
    vocabulary = load_file(vocab_filename)
    return set(vocabulary.split())


def ulterior_token_clean(tweets_tokens, vocab_filename, filtered_tokens_filename):
    if not os.path.exists(filtered_tokens_filename):
        lemmatizer = WordNetLemmatizer()
        filtered_tweets_tokens = []
        for tweet_tokens in tweets_tokens:
            filtered_tweet_tokens = []
            for token in tweet_tokens.split():
                filtered_token = lemmatizer.lemmatize(token.lower())
                filtered_token = lemmatizer.lemmatize(filtered_token, 'v')
                filtered_tweet_tokens.append(filtered_token)
            filtered_tweets_tokens.append(filtered_tweet_tokens)
        vocab = build_vocabulary(vocab_filename, filtered_tweets_tokens, minimum_occurrence=5)
        filtered_again = []
        for tweet in filtered_tweets_tokens:
            filtered_again.append(' '.join([w for w in tweet if w in vocab]))
        save_file(filtered_again, filtered_tokens_filename)
    # Load the filtered tokens
    filtered_tweets_tokens = load_file(filtered_tokens_filename).split("\n")
    return filtered_tweets_tokens


def ulterior_pos_clean(tweets_pos_tags, vocab_filename, filtered_pos_filename):
    if not os.path.exists(filtered_pos_filename):
        vocab = build_vocabulary(vocab_filename, tweets_pos_tags, minimum_occurrence=50)
        filtered_tweets_pos_tags = []
        for tweet_pos_tags in tweets_pos_tags:
            filtered_tweets_pos_tags.append(' '.join([pos_tag for pos_tag in tweet_pos_tags if pos_tag in vocab]))
        save_file(filtered_tweets_pos_tags, filtered_pos_filename)
    # Load the filtered tokens
    filtered_tweets_pos_tags = load_file(filtered_pos_filename).split("\n")
    return filtered_tweets_pos_tags


def get_tags_for_each_tweet(tweets_filename, tokens_filename, pos_filename):
    if not os.path.exists(pos_filename):
        tweets = load_file(tweets_filename).split("\n")
        tokens_lines = []
        pos_lines = []
        tokens_line = ""
        pos_line = ""
        for t in tweets:
            if len(t) < 1:
                tokens_lines.append(tokens_line[:])
                pos_lines.append(pos_line[:])
                tokens_line = ""
                pos_line = ""
            else:
                t_split = t.split("\t")
                tokens_line += t_split[0] + " "
                pos_line += t_split[1] + " "
        save_file(tokens_lines, tokens_filename)
        save_file(pos_lines, pos_filename)
    # Load the tokens and the pos for the tweets in this set
    tokens = load_file(tokens_filename).split("\n")
    pos = load_file(pos_filename).split("\n")
    return tokens, pos


# Initial tweet cleaning - useful to filter data before tokenization
def clean_tweet(tweet, word_list):
    # Add white space before every punctuation sign so that we can split around it and keep it
    tweet = re.sub('([!?*&%"~`^+{}])', r' \1 ', tweet)
    tweet = re.sub('\s{2,}', ' ', tweet)
    tokens = tweet.split()
    valid_tokens = []
    for word in tokens:
        if word.lower().startswith('#sarca'):           # do not include any #sarca* hashtags
            continue
        if word.startswith('@'):                        # replace all mentions with a general user
            word = '@user'
        if word.startswith('http'):                     # do not include URLs
            continue
        if word.startswith('#'):                        # split hash tag
            word = splitter.split_hashtag(word[1:], word_list)
            valid_tokens.extend(word)
            continue
        valid_tokens.append(word)
    return ' '.join(valid_tokens)


def process_tweets(data, word_list):
    tweets = []
    for tweet in data:
        clean_tw = clean_tweet(tweet, word_list)
        tweets.append(clean_tw)
    return tweets


def initial_clean(tweets, clean_filename, word_file):
    if not os.path.exists(clean_filename):
        word_file = path + "/res/" + word_file
        word_list = load_file(word_file).split()
        filtered_tweets = process_tweets(tweets, word_list=word_list)
        save_file(filtered_tweets, clean_filename)


def get_clean_data(train_filename, test_filename, word_list):
    tweets_filename = "tweets_"
    tokens_filename = "tokens_"
    pos_filename = "pos_"
    tokens_vocab_filename = "vocabulary_of_tokens_" + train_filename
    pos_vocab_filename = "vocabulary_of_pos_tags_" + train_filename

    # Load the train and test sets
    print("Loading data...")
    train_tweets, train_labels = load_data_panda(path + "/res/" + train_filename)
    test_tweets, test_labels = load_data_panda(path + "/res/" + test_filename)

    # Initial clean of data
    initial_clean(train_tweets, path + "/res/clean_" + train_filename, word_list)
    initial_clean(test_tweets, path + "/res/clean_" + test_filename, word_list)

    # Get the tags corresponding to the test and train files
    train_tokens, train_pos = get_tags_for_each_tweet(path + "/res/" + tweets_filename + train_filename,
                                                      path + "/res/" + tokens_filename + train_filename,
                                                      path + "/res/" + pos_filename + train_filename)
    test_tokens, test_pos = get_tags_for_each_tweet(path + "/res/" + tweets_filename + test_filename,
                                                      path + "/res/" + tokens_filename + test_filename,
                                                      path + "/res/" + pos_filename + test_filename)
    # Ulterior clean of the tags (tokens and pos)
    
    filtered_train_tokens = ulterior_token_clean(train_tokens, path + "/res/" + tokens_vocab_filename,
                                                 path + "/res/filtered_" + tokens_filename + train_filename)
    filtered_test_tokens = ulterior_token_clean(test_tokens, path + "/res/" + tokens_vocab_filename,
                                                path + "/res/filtered_" + tokens_filename + test_filename)
    filtered_train_pos = ulterior_pos_clean(train_pos, path + "/res/" + pos_vocab_filename,
                                            path + "/res/filtered_" + pos_filename + train_filename)
    filtered_test_pos = ulterior_pos_clean(test_pos, path + "/res/" + pos_vocab_filename,
                                           path + "/res/filtered_" + pos_filename + test_filename)
    
    return train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels, \
           test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels
