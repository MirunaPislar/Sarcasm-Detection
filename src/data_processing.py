import re
import os
import numpy as np
from pandas import read_csv
from collections import Counter
from nltk.tokenize import TweetTokenizer


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


def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_data_panda(filename):
    print("Reading data from file %s..." % filename)
    data = read_csv(filename, sep="\t+", header=None, engine='python')
    data.columns = ["Set", "Label", "Text"]
    print('The shape of the train set is: ', data.shape)
    print('Columns: ', data.columns.values)
    return np.array(data["Text"]), np.array(data["Label"])


def build_subj_dicionary(lines):
    subj_dict = dict()
    lines = lines.split("\n")
    for line in lines:
        splits = line.split(' ')
        if len(splits) == 6:
            word = splits[2][6:]        # the word analyzed
            word_type = splits[0][5:]   # weak or strong subjective
            pos = splits[3][5:]         # part of speech: noun, verb, adj, adv and anypos
            polarity = splits[5][14:]   # its polarity: can be positive, negative or neutral
            new_dict_entry = {pos: [word_type, polarity]}
            if word in subj_dict.keys():
                subj_dict[word].update(new_dict_entry)
            else:
                subj_dict[word] = new_dict_entry
    return subj_dict


def get_subj_lexicon(path):
    filename = path + "/res/subjectivity_lexicon.tff"
    lexicon = load_file(filename)
    subj_dict = build_subj_dicionary(lexicon)
    return subj_dict


def get_emoji_dictionary(path):
    filename = path + "/res/emoji_list.txt"
    emojis = load_file(filename).split("\n")
    emoji_dict = {}
    for line in emojis:
        line = line.split(" ", 1)
        emoji = line[0]
        description = line[1]
        emoji_dict[emoji] = description
    return emoji_dict


def build_emoji_sentiment_dictionary(path):
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
                         + "\t" + str(positive) + "\t" + description)
        save_file(lines, new_emoji_sentiment_filename)
    emoji_sentiment_data = load_file(new_emoji_sentiment_filename).split("\n")
    emoji_sentiment_dict = {}
    for line in emoji_sentiment_data:
        line = line.split("\t")
        # Get emoji characteristics: negative neutral positive description
        emoji_sentiment_dict[line[0]] = [line[1], line[2], line[3], line[4]]
    return emoji_sentiment_dict


def get_pos_file(filename, to_save_filename):
    pos_tweets = load_file(filename).split("\n")
    lines = []
    line = ""
    for p in pos_tweets:
        if len(p) < 1:
            lines.append(line[:])
            line = ""
        else:
            line += p.split("\t")[1] + " "
    save_file(lines, to_save_filename)


def get_pos_tags_for_each_tweet(path, filename):
    # Build the semantic corpus if not already there
    if not os.path.exists(path + "/res/pos_" + filename):
        tweets_pos_filename = path + "/res/tweets_pos_" + filename
        to_save_filename = path + "/res/pos_" + filename
        get_pos_file(tweets_pos_filename, to_save_filename)
    # Count each part of speech tag and register the counts in a dictionary
    semantic_data = load_file(path + "/res/pos_" + filename).split("\n")
    return semantic_data


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
    if not os.path.exists(vocab_filename):
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
        save_file(vocabulary.keys(), vocab_filename)
        print("Vocabulary saved to file \"%s\"" % vocab_filename)
        print("The top 50 most common words: ", vocabulary.most_common(50))
    vocabulary = load_file(vocab_filename)
    return set(vocabulary.split())


def save_file(lines, filename):
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
