import re, os
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import words
import utils


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
    lexicon = utils.load_file(filename)
    subj_dict = build_subj_dicionary(lexicon)
    return subj_dict


def get_emoji_dictionary():
    filename = path + "/res/emoji_list.txt"
    emojis = utils.load_file(filename).split("\n")
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
        emojis = utils.load_file(filename).split("\n")[1:]
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
            utils.save_file(lines, new_emoji_sentiment_filename)
    emoji_sentiment_data = utils.load_file(new_emoji_sentiment_filename).split("\n")
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
            vocabulary.update([l.lower() for l in line])
        print("The top 10 most common words: ", vocabulary.most_common(10))
        # Filter all words that appear too rarely or too frequent to be conclusive
        vocabulary = {key: vocabulary[key] for key in vocabulary
                      if vocabulary[key] >= minimum_occurrence
                      and key not in vocabulary.most_common(10)}
        utils.save_file(vocabulary.keys(), vocab_filename)
        print("Vocabulary saved to file \"%s\"" % vocab_filename)
    vocabulary = set(utils.load_file(vocab_filename).split())
    print("Loaded vocabulary of size ", len(vocabulary))
    return vocabulary


def build_vocabulary_for_dnn_tasks(vocab_filename, lines):
    if not os.path.exists(vocab_filename):
        print("Building vocabulary...")
        vocabulary = Counter()
        for line in lines:
            vocabulary.update([l.lower() for l in line])
        vocabulary = {key: vocabulary[key] for key in vocabulary}
        vocabulary = sorted(vocabulary.items(), key=lambda pair: pair[1], reverse=True)
        counter = 1
        indexed_vocabulary = {}
        for (key, _) in vocabulary:
            indexed_vocabulary[key] = counter
            counter += 1
        indexed_vocabulary['unk'] = len(indexed_vocabulary) + 1
        utils.save_dictionary(indexed_vocabulary, vocab_filename)
        print("Vocabulary saved to file \"%s\"" % vocab_filename)
    vocabulary = utils.load_dictionary(vocab_filename)
    print("Loaded vocabulary of size ", len(vocabulary))
    return vocabulary


def vocabulary_filtering(vocabulary, lines):
    filtered_lines = []
    indices = []
    for line in lines:
        filtered_line = []
        individual_word_indices = []
        for word in line:
            word = word.lower()
            if word in vocabulary:
                individual_word_indices.append(vocabulary[word])
                filtered_line.append(word)
            else:
                individual_word_indices.append(vocabulary['unk'])
                filtered_line.append('unk')
        indices.append(individual_word_indices)
        filtered_lines.append(filtered_line)
    return filtered_lines, indices


# Extract the lemmatized nouns and/or verbs from a set of documents - used in LDA modelling
def extract_lemmatized_tweet(tokens, pos, use_verbs=True, use_nouns=True, use_all=False):
    lemmatizer = WordNetLemmatizer()
    clean_data = []
    for index in range(len(tokens)):
        if use_verbs and pos[index] is 'V':
            clean_data.append(lemmatizer.lemmatize(tokens[index].lower(), 'v'))
        if use_nouns and pos[index] is 'N':
            clean_data.append(lemmatizer.lemmatize(tokens[index].lower()))
        if use_all:
            lemmatized_word = lemmatizer.lemmatize(tokens[index].lower(), 'v')
            word = lemmatizer.lemmatize(lemmatized_word)
            if pos[index] not in ['^', ',', '$', '&', '!', '#', '@']:
                clean_data.append(word)
    return clean_data


# Split a long, compound hash tag into its component tags. Given the character limit of tweets,
# people would stick words together to save space so this is a useful tool.
# Examples of hash splits from real data (train set) are in /stats/hashtag_splits.txt
# Implementation adapted from https://github.com/matchado/HashTagSplitter
def split_hashtag_to_words_all_possibilities(hashtag, word_dictionary):
    all_possibilities = []
    split_possibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag) + 1))]
    possible_split_positions = [i for i, x in enumerate(split_possibility) if x is True]

    for split_pos in possible_split_positions:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag) - split_pos], hashtag[len(hashtag) - split_pos:]

        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)
            another_round = split_hashtag_to_words_all_possibilities(word_2, word_dictionary)
            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
        else:
            another_round = split_hashtag_to_words_all_possibilities(word_2, word_dictionary)
            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
    return all_possibilities


def split_hashtag(hashtag, wordlist):
    split_words = []
    if hashtag != hashtag.lower() and hashtag != hashtag.upper():
        split_words = re.findall('[A-Z][^A-Z]*', hashtag)
    else:
        j = 0
        while j <= len(hashtag):
            loc = j
            for i in range(j + 1, len(hashtag) + 1, 1):
                if hashtag[j:i].lower() in wordlist:
                    loc = i
            if loc == j:
                j += 1
            else:
                split_words.append(hashtag[j:loc])
                j = loc
    split_words = ['#' + str(s) for s in split_words]
    return split_words


# Select the best possible hashtag split based on upper-case
# or component words maximizing the length of the possible word split
def split_hashtag_long_version(hashtag):
    path = os.getcwd()[:os.getcwd().rfind('/')]
    word_file = path + "/res/word_list.txt"
    word_list = utils.load_file(word_file).split()
    word_dictionary = list(set(words.words()))
    for alphabet in "bcdefghjklmnopqrstuvwxyz":
        word_dictionary.remove(alphabet)
    all_poss = split_hashtag_to_words_all_possibilities(hashtag.lower(), word_dictionary)
    max_p = 0
    min_len = 1000
    found = False
    best_p = []
    for poss in all_poss:
        counter = 0
        for p in poss:
            if p in word_list:
                counter += 1
        if counter == len(poss) and min_len > counter:
            found = True
            min_len = counter
            best_p = poss
        else:
            if counter > max_p and not found:
                max_p = counter
                best_p = poss
    best_p_v2 = split_hashtag(hashtag, word_list)
    if best_p != [] and best_p_v2 != []:
        split_words = best_p if len(best_p) < len(best_p_v2) else best_p_v2
    else:
        if best_p == [] and best_p_v2 == []:
            split_words = [hashtag]
        else:
            split_words = best_p if best_p_v2 == [] else best_p_v2
    split_words = ['#' + str(s) for s in split_words]
    return split_words


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
            filtered_again_tweet = []
            for word in tweet:
                if word in vocab:
                    filtered_again_tweet.append(word)
                else:
                    filtered_again_tweet.append('unk')
            filtered_again.append(' '.join([f for f in filtered_again_tweet]))
        utils.save_file(filtered_again, filtered_tokens_filename)
    # Load the filtered tokens
    filtered_tweets_tokens = utils.load_file(filtered_tokens_filename).split("\n")
    return filtered_tweets_tokens


def ulterior_pos_clean(tweets_pos_tags, vocab_filename, filtered_pos_filename):
    if not os.path.exists(filtered_pos_filename):
        vocab = build_vocabulary(vocab_filename, tweets_pos_tags, minimum_occurrence=50)
        filtered_tweets_pos_tags = []
        for tweet_pos_tag in tweets_pos_tags:
            # filtered_tweets_pos_tags.append(' '.join([pos_tag for pos_tag in tweet_pos_tag if pos_tag in vocab]))
            filtered_again_pos_tags = []
            for pos in tweet_pos_tag:
                if pos in vocab:
                    filtered_again_pos_tags.append(pos)
                else:
                    filtered_again_pos_tags.append('unk')
            filtered_tweets_pos_tags.append(' '.join([f for f in filtered_again_pos_tags]))
        utils.save_file(filtered_tweets_pos_tags, filtered_pos_filename)
    # Load the filtered tokens
    filtered_tweets_pos_tags = utils.load_file(filtered_pos_filename).split("\n")
    return filtered_tweets_pos_tags


def get_tags_for_each_tweet(tweets_filename, tokens_filename, pos_filename):
    if not os.path.exists(pos_filename):
        tweets = utils.load_file(tweets_filename).split("\n")
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
        utils.save_file(tokens_lines, tokens_filename)
        utils.save_file(pos_lines, pos_filename)
    # Load the tokens and the pos for the tweets in this set
    tokens = utils.load_file(tokens_filename).split("\n")
    pos = utils.load_file(pos_filename).split("\n")
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
            word = split_hashtag(word[1:], word_list)
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


def process_set(dataset_filename, vocab_filename, word_list):
    data, labels = utils.load_data_panda(dataset_filename)
    tweets = process_tweets(data, word_list)
    vocabulary = build_vocabulary(tweets, vocab_filename, minimum_occurrence=10)
    filtered_tweets = []
    for tweet in tweets:
        filtered_tweets.append([t for t in tweet if t in vocabulary])
    return filtered_tweets, labels


def initial_clean(tweets, clean_filename, word_file):
    if not os.path.exists(clean_filename):
        word_file = path + "/res/" + word_file
        word_list = utils.load_file(word_file).split()
        filtered_tweets = process_tweets(tweets, word_list=word_list)
        utils.save_file(filtered_tweets, clean_filename)
        return filtered_tweets
    else:
        filtered_tweets = [[word for word in line.split()] for line in utils.load_file(clean_filename).split("\n")]
        return filtered_tweets


def get_clean_data(train_filename, test_filename, word_list):
    tweets_filename = "tweets_"
    tokens_filename = "tokens_"
    pos_filename = "pos_"
    tokens_vocab_filename = "vocabulary_of_tokens_" + train_filename
    pos_vocab_filename = "vocabulary_of_pos_tags_" + train_filename

    # Load the train and test sets
    print("Loading data...")
    train_tweets, train_labels = utils.load_data_panda(path + "/res/" + train_filename)
    test_tweets, test_labels = utils.load_data_panda(path + "/res/" + test_filename)

    # Initial clean of data
    initial_clean(train_tweets, path + "/res/clean_" + train_filename, word_list)
    initial_clean(test_tweets, path + "/res/clean_" + test_filename, word_list)

    # Get the tags corresponding to the test and train files
    train_tokens, train_pos = \
        get_tags_for_each_tweet(path + "/res/" + tweets_filename + train_filename,
                                path + "/res/" + tokens_filename + train_filename,
                                path + "/res/" + pos_filename + train_filename)
    test_tokens, test_pos = \
        get_tags_for_each_tweet(path + "/res/" + tweets_filename + test_filename,
                                path + "/res/" + tokens_filename + test_filename,
                                path + "/res/" + pos_filename + test_filename)
    # Ulterior clean of the tags (tokens and pos)
    
    filtered_train_tokens = \
        ulterior_token_clean(train_tokens, path + "/res/" + tokens_vocab_filename,
                             path + "/res/filtered_" + tokens_filename + train_filename)
    filtered_test_tokens = \
        ulterior_token_clean(test_tokens, path + "/res/" + tokens_vocab_filename,
                             path + "/res/filtered_" + tokens_filename + test_filename)
    filtered_train_pos = \
        ulterior_pos_clean(train_pos, path + "/res/" + pos_vocab_filename,
                           path + "/res/filtered_" + pos_filename + train_filename)
    filtered_test_pos = \
        ulterior_pos_clean(test_pos, path + "/res/" + pos_vocab_filename,
                           path + "/res/filtered_" + pos_filename + test_filename)
    
    return train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,\
        test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels


def get_clean_dl_data(train_filename, dev_filename, test_filename, word_list):
    vocab_filename = "dnn_vocabulary_" + train_filename

    # Load the train and test sets
    print("Loading data...")
    train_tweets, train_labels = utils.load_data_panda(path + "/res/" + train_filename)
    dev_tweets, dev_labels = utils.load_data_panda(path + "/res/" + dev_filename)
    test_tweets, test_labels = utils.load_data_panda(path + "/res/" + test_filename)

    # Initial clean of data
    clean_train_tweets = initial_clean(train_tweets, path + "/res/clean_" + train_filename, word_list)
    clean_dev_tweets = initial_clean(dev_tweets, path + "/res/clean_" + dev_filename, word_list)
    clean_test_tweets = initial_clean(test_tweets, path + "/res/clean_" + test_filename, word_list)

    vocabulary = build_vocabulary_for_dnn_tasks(path + "/res/" + vocab_filename, clean_train_tweets)
    clean_train_tweets, train_indices = vocabulary_filtering(vocabulary, clean_train_tweets)
    clean_dev_tweets, dev_indices = vocabulary_filtering(vocabulary, clean_dev_tweets)
    clean_test_tweets, test_indices = vocabulary_filtering(vocabulary, clean_test_tweets)

    return clean_train_tweets, train_indices, train_labels, \
        clean_dev_tweets, dev_indices, dev_labels, \
        clean_test_tweets, test_indices, test_labels, len(vocabulary)
