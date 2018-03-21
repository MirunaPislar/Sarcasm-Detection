from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import ngrams, pos_tag
from collections import Counter
import numpy as np
import vocab_helpers as helper


def count_apparitions(tokens, list_to_count_from):
    total_count = 0.0
    for affirmative in list_to_count_from:
        total_count += tokens.count(affirmative)
    return total_count


def get_features1(tweets, subj_dict):
    print("Getting features type 1...")
    features = []
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    lemmatizer = WordNetLemmatizer()
    # Take positive and negative noun/verb phrases as features
    for tweet in tweets:
        feature_list = [0.0] * 6
        tokens = tknzr.tokenize(tweet)
        pos = pos_tag(tokens)
        pos = [p for p in pos if 'VB' in p[1] or 'NN' in p[1]]
        for p in pos:
            stemmed = lemmatizer.lemmatize(p[0], 'v')
            stemmed = lemmatizer.lemmatize(stemmed)
            if 'VB' in p[1] and stemmed in subj_dict:
                if 'verb' in subj_dict[stemmed]:
                    if 'positive' in subj_dict[stemmed]['verb']:
                        feature_list[0] += 1.0
                    if 'negative' in subj_dict[stemmed]['verb']:
                        feature_list[1] += 1.0
                elif 'anypos' in subj_dict[stemmed]:
                    if 'positive' in subj_dict[stemmed]['anypos']:
                        feature_list[0] += 1.0
                    if 'negative' in subj_dict[stemmed]['anypos']:
                        feature_list[1] += 1.0
            if 'NN' in p[1] and stemmed in subj_dict:
                if 'noun' in subj_dict[stemmed]:
                    if 'positive' in subj_dict[stemmed]['noun']:
                        feature_list[2] += 1.0
                    if 'negative' in subj_dict[stemmed]['noun']:
                        feature_list[3] += 1.0
                elif 'anypos' in subj_dict[stemmed]:
                    if 'positive' in subj_dict[stemmed]['anypos']:
                        feature_list[2] += 1.0
                    if 'negative' in subj_dict[stemmed]['anypos']:
                        feature_list[3] += 1.0
        # Derive features from punctuation
        feature_list[4] += count_apparitions(tokens, helper.punctuation)
        # Take the number of strong negations as a feature
        feature_list[5] += count_apparitions(tokens, helper.strong_negations)
        features.append(feature_list)
    print("Done.")
    return features


def get_features2(tweets, subj_dict):
    print("Getting features type 2...")
    features = []
    tknzr = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
    lemmatizer = WordNetLemmatizer()
    for tweet in tweets:
        feature_list = [0.0] * 5
        tokens = tknzr.tokenize(tweet)
        # Take the number of positive and negative words as features
        for word in tokens:
            stemmed = lemmatizer.lemmatize(word, 'v')
            stemmed = lemmatizer.lemmatize(stemmed)
            if stemmed in subj_dict:
                dictlist = []
                for word in subj_dict[stemmed]:
                    dictlist.extend(subj_dict[stemmed][word])
                if 'strongsubj' in dictlist:
                    value = 1.0
                else:
                    value = 0.5
                if 'positive' in dictlist:
                    feature_list[0] += value
                elif 'negative' in dictlist:
                    feature_list[1] += value
        # Take the report of positives to negatives as a feature
        if feature_list[0] != 0.0 and feature_list[1] != 0.0:
            feature_list[2] = feature_list[0] / feature_list[1]
        # Derive features from punctuation
        feature_list[2] += count_apparitions(tokens, helper.punctuation)
        # Take strong negations as a feature
        feature_list[3] += count_apparitions(tokens, helper.strong_negations)
        # Take strong affirmatives as a feature
        feature_list[4] += count_apparitions(tokens, helper.strong_affirmatives)
        features.append(feature_list)
    print("Done.")
    return features


def get_features3(tweets, subj_dict):
    print("Getting features type 3...")
    features = []
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    lemmatizer = WordNetLemmatizer()
    # Take positive and negative noun/verb phrases as features
    for tweet in tweets:
        feature_list = [0.0] * 8
        tokens = tknzr.tokenize(tweet)
        pos = pos_tag(tokens)
        pos = [p for p in pos if 'VB' in p[1] or 'NN' in p[1]]
        for p in pos:
            stemmed = lemmatizer.lemmatize(p[0], 'v')
            stemmed = lemmatizer.lemmatize(stemmed)
            if 'VB' in p[1] and stemmed in subj_dict:
                if 'verb' in subj_dict[stemmed]:
                    if 'strongsubj' in subj_dict[stemmed]['verb']:
                        value = 1.0
                    else:
                        value = 0.5
                    if 'positive' in subj_dict[stemmed]['verb']:
                        feature_list[0] += value
                    elif 'negative' in subj_dict[stemmed]['verb']:
                        feature_list[1] += value
                elif 'anypos' in subj_dict[stemmed]:
                    if 'strongsubj' in subj_dict[stemmed]['anypos']:
                        value = 1.0
                    else:
                        value = 0.5
                    if 'positive' in subj_dict[stemmed]['anypos']:
                        feature_list[0] += value
                    elif 'negative' in subj_dict[stemmed]['anypos']:
                        feature_list[1] += value
            if 'NN' in p[1] and stemmed in subj_dict:
                if 'noun' in subj_dict[stemmed]:
                    if 'strongsubj' in subj_dict[stemmed]['noun']:
                        value = 1.0
                    else:
                        value = 0.5
                    if 'positive' in subj_dict[stemmed]['noun']:
                        feature_list[2] += value
                    elif 'negative' in subj_dict[stemmed]['noun']:
                        feature_list[3] += value
                elif 'anypos' in subj_dict[stemmed]:
                    if 'strongsubj' in subj_dict[stemmed]['anypos']:
                        value = 1.0
                    else:
                        value = 0.5
                    if 'positive' in subj_dict[stemmed]['anypos']:
                        feature_list[2] += value
                    elif 'negative' in subj_dict[stemmed]['anypos']:
                        feature_list[3] += value
        # Take the report of positives to negatives as a feature
        if (feature_list[0] + feature_list[2]) != 0.0 and (feature_list[1] + feature_list[3]) != 0.0:
            feature_list[4] = (feature_list[0] + feature_list[2]) / (feature_list[1] + feature_list[3])
        # Derive features from punctuation
        feature_list[5] += count_apparitions(tokens, helper.punctuation)
        # Take strong negations as a feature
        feature_list[6] += count_apparitions(tokens, helper.strong_negations)
        # Take strong affirmatives as a feature
        feature_list[7] += count_apparitions(tokens, helper.strong_affirmatives)
        features.append(feature_list)
    print("Done.")
    return features


def get_ngram_list(tknzr, text, n):
    tokens = tknzr.tokenize(text)
    tokens = [t for t in tokens if not t.startswith('#')]
    tokens = [t for t in tokens if not t.startswith('@')]
    ngram_list = [gram for gram in ngrams(tokens, n)]
    return ngram_list


def get_ngrams(tweets, n):
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    regexp_tknzr = RegexpTokenizer(r'\w+')
    tweet_tknzr = TweetTokenizer()
    for tweet in tweets:
        tweet = tweet.lower()
        # Get the unigram list for this tweet and update the unigram counter
        unigram_list = get_ngram_list(tweet_tknzr, tweet, 1)
        unigrams.update(unigram_list)
        # Get the bigram list for this tweet and update the bigram counter
        if n > 1:
            bigram_list = get_ngram_list(regexp_tknzr, tweet, 2)
            bigrams.update(bigram_list)
            # Get the trigram list for this tweet and update the trigram counter
            if n > 2:
                trigram_list = get_ngram_list(regexp_tknzr, tweet, 3)
                trigrams.update(trigram_list)
    # Update the counters such that each n-gram appears at least min_occurence times
    min_occurence = 2
    unigram_tokens = [k for k, c in unigrams.items() if c >= min_occurence]
    # In case using just unigrams, make the bigrams and trigrams empty
    bigram_tokens = trigram_tokens = []
    if n > 1:
        bigram_tokens = [k for k, c in bigrams.items() if c >= min_occurence]
    if n > 2:
        trigram_tokens = [k for k, c in trigrams.items() if c >= min_occurence]
    return unigram_tokens, bigram_tokens, trigram_tokens


def create_ngram_mapping(unigrams, bigrams, trigrams):
    ngram_map = dict()
    all_ngrams = unigrams
    all_ngrams.extend(bigrams)
    all_ngrams.extend(trigrams)
    for i in range(0, len(all_ngrams)):
        ngram_map[all_ngrams[i]] = i
    return ngram_map


def get_ngram_features_from_map(tweets, ngram_map, n):
    regexp_tknzr = RegexpTokenizer(r'\w+')
    tweet_tknzr = TweetTokenizer()
    features = []
    for tweet in tweets:
        feature_list = [0] * np.zeros(len(ngram_map))
        tweet = tweet.lower()
        ngram_list = get_ngram_list(tweet_tknzr, tweet, 1)
        if n > 1:
            ngram_list += get_ngram_list(regexp_tknzr, tweet, 2)
        if n > 2:
            ngram_list += get_ngram_list(regexp_tknzr, tweet, 3)
        for gram in ngram_list:
            if gram in ngram_map:
                feature_list[ngram_map[gram]] += 1.0
        features.append(feature_list)
    return features


def get_ngram_features(tweets, n):
    print("Getting n-gram features...")
    unigrams = []
    bigrams = []
    trigrams = []
    if n == 1:
        unigrams, _, _ = get_ngrams(tweets, n)
    if n == 2:
        unigrams, bigrams, _ = get_ngrams(tweets, n)
    if n == 3:
        unigrams, bigrams, trigrams = get_ngrams(tweets, n)
    ngram_map = create_ngram_mapping(unigrams, bigrams, trigrams)
    features = get_ngram_features_from_map(tweets, ngram_map, n)
    print("Done.")
    return ngram_map, features
