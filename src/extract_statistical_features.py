import emoji
import re
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import  PorterStemmer, ngrams
from nltk.stem.wordnet import WordNetLemmatizer
import data_processing as data_proc


# Get 12 pragmatic features (like the presence of laughter, capitalized words, emojis, user mentions)
# but also counts of punctuation, strong affirmatives, negations, intensifiers, tokens and average token size
def get_pragmatic_features(tweet, tweet_tokens):
    tweet_len_ch = len(tweet)               # get the length of the tweet in characters
    tweet_len_tokens = len(tweet_tokens)    # get the length of the tweet in tokens
    average_token_length = len(tweet) * 1.0 / len(tweet_tokens)     # average tweet length
    capitalized_words = laughter = user_mentions = negations = affirmatives \
        = interjections = intensifiers = punctuation = emojis = 0
    for t in tweet:
        if t.isupper():
            capitalized_words = 1   # binary feature marking the presence of capitalized words
        if t.startswith("@"):
            user_mentions = 1
    for t in tweet_tokens:
        if t.startswith("haha") or re.match('l(o)+l$', t):
            laughter = 1        # binary feature marking the presence of laughter
        if t in data_proc.strong_negations:
            negations += 1      # count-based feature of strong negations
        if t in data_proc.strong_affirmatives:
            affirmatives += 1   # count-based feature of strong affirmatives
        if t in data_proc.interjections:
            interjections += 1  # count-based feature of relevant interjections
        if t in data_proc.intensifiers:
            intensifiers += 1   # count-based feature of relevant intensifiers
        if t in data_proc.punctuation:
            punctuation += 1    # count-based feature of relevant punctuation signs
        if t in emoji.UNICODE_EMOJI:
            emojis += 1         # count-based feature of relevant punctuation signs
    feature_list = {'tw_len_ch': tweet_len_ch, 'tw_len_tok': tweet_len_tokens, 'avg_len': average_token_length,
                    'capitalized': capitalized_words, 'laughter': laughter, 'user_mentions': user_mentions,
                    'negations': negations, 'affirmatives': affirmatives, 'interjections': interjections,
                    'intensifiers': intensifiers, 'punctuation': punctuation, 'emojis': emojis}
    return feature_list


# Obtain 25 POS Tags
def get_semantic_features(pos_tweet):
    pos_tweet = pos_tweet.split()
    # CMU Twitter Part-of-Speech Tagger according to the following paper by Gimpel et al.
    # "Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments"
    pos_dict = dict.fromkeys(['N', 'O', 'S', '^', 'Z', 'L', 'M', 'V', 'A', 'R', '!', 'D', 'P',
                              '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E', '$', ',', 'G'], 0)
    for pos in pos_tweet:
        pos_dict[pos] += 1
    return pos_dict


# Obtain a n-grams (specified as a list n = [1, 2, 3, ...]) for any kind of tokens
# i.e both words and pos tags
def get_ngrams(tweet, tokens, n, use_just_words=False):
    if len(n) < 1:
        return {}
    if use_just_words:
        word_tokenizer = RegexpTokenizer(r'\w+')
        tokens = word_tokenizer.tokenize(tweet)
    porter = PorterStemmer()
    tokens = [porter.stem(t.lower()) for t in tokens]
    ngram_tokens = []
    for i in n:
        for gram in ngrams(tokens, i):
            string_token = 'gram '
            for j in range(i):
                string_token += gram[j] + ' '
            ngram_tokens.append(string_token)
    ngram_features = {i: ngram_tokens.count(i) for i in set(ngram_tokens)}
    return ngram_features


# Get sentiment features -- a total of 14 features derived
# Emoji features: a count of the positive, negative and neutral emojis
# along with the ratio of positive to negative emojis and negative to neutral
# Using the MPQA subjectivity lexicon, we have to check words for their pos,
# and obtain features: a count of positive, negative and neutral words, as well as
# a count of the strong and weak subjectives, along with their ratios and a total sentiment words
# Also using VADER sentiment analyser to obtain a score of sentiments held in a tweet
def get_sentiment_features(path, tweet, tokens, pos_tweet):
    sent_features = dict.fromkeys(["positive emoji", "negative emoji", "neutral emoji",
                                   "emojis pos : neg", "emojis neg : neutral",
                                   "subjlexicon weaksubj", "subjlexicon strongsubj",
                                   "subjlexicon positive", "subjlexicon negative",
                                   "subjlexicon neutral", "words pos : neg", "words neg : neutral",
                                   "subjectivity strong : weak", "total sentiment words"], 0.0)

    # Emoji lexicon - underlying sentiment (pos, neutral, neg)
    emoji_sent_dict = data_proc.build_emoji_sentiment_dictionary(path)
    for t in tokens:
        if t in emoji_sent_dict.keys():
            sent_features['negative emoji'] += float(emoji_sent_dict[t][0])
            sent_features['neutral emoji'] += float(emoji_sent_dict[t][1])
            sent_features['positive emoji'] += float(emoji_sent_dict[t][2])
    if sent_features['negative emoji'] != 0:
        sent_features['emojis pos : neg'] = sent_features['positive emoji'] \
                                            / float(sent_features['negative emoji'])
    if sent_features['neutral emoji'] != 0:
        sent_features['emojis neg : neutral'] = sent_features['negative emoji'] \
                                            / float(sent_features['neutral emoji'])

    # Obtain subjectivity features from the MPQA lexicon
    # Build the subjectivity lexicon, if not already there
    subj_dict = data_proc.get_subj_lexicon(path)
    lemmatizer = WordNetLemmatizer()
    pos_translation = {'N': 'noun', 'V': 'verb', 'D': 'adj', 'R': 'adverb'}
    for index in range(len(tokens)):
        stemmed = lemmatizer.lemmatize(tokens[index], 'v')
        if stemmed in subj_dict.keys():
            if pos_tweet[index] in pos_translation and pos_translation[pos_tweet[index]] in subj_dict[stemmed].keys():
                sent_features['subjlexicon ' + subj_dict[stemmed][pos_translation[pos_tweet[index]]][0]] += 1
                sent_features['subjlexicon ' + subj_dict[stemmed][pos_translation[pos_tweet[index]]][1]] += 1
            else:
                if 'anypos' in subj_dict[stemmed].keys():
                    sent_features['subjlexicon ' + subj_dict[stemmed]['anypos'][0]] += 1
                    sent_features['subjlexicon ' + subj_dict[stemmed]['anypos'][1]] += 1
    sent_features["total sentiment words"] = sent_features["subjlexicon positive"] \
                                             + sent_features["subjlexicon negative"] \
                                             + sent_features["subjlexicon neutral"]
    if sent_features["subjlexicon negative"] != 0:
        sent_features["words pos : neg"] = sent_features["subjlexicon positive"] \
                                           / float(sent_features["subjlexicon negative"])
    if sent_features["subjlexicon neutral"] != 0:
        sent_features["words neg : neutral"] = sent_features["subjlexicon negative"] \
                                           / float(sent_features["subjlexicon neutral"])
    if sent_features["subjlexicon strongsubj"] != 0:
        sent_features["subjectivity strong : weak"] = sent_features["subjlexicon weaksubj"] \
                                           / float(sent_features["subjlexicon strongsubj"])

    # Vader Sentiment Analyser
    sia = SentimentIntensityAnalyzer()
    polarity_scores = sia.polarity_scores(tweet)
    for name, score in polarity_scores.items():
        sent_features["vader score " + name] = score
    return sent_features


# Collect all features
def get_feature_set(tweets, path, filename, pragmatic=True, semantic_unigrams=True, semantic_bigrams=True,
                    lexical=True, ngram_list=[], sentiment=True):
    features = []
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)
    filename = filename.split("/")[len(filename.split("/")) - 1]

    # Obtain the semantic vocabulary, if necessary
    pos_tweets = data_proc.get_pos_tags_for_each_tweet(path, filename)
    index = 0
    for tweet in tweets:
        tweet_tokens = tknzr.tokenize(tweet)
        tweet_tokens = [t for t in tweet_tokens if not t.startswith("#sarca")]
        pos_this_tweet = pos_tweets[index].split()
        pragmatic_features = semantic_features = pos_bigrams = words_ngrams = sentiment_features = {}
        if pragmatic:
            pragmatic_features = get_pragmatic_features(tweet, tweet_tokens)
        if semantic_unigrams:
            semantic_features = get_semantic_features(pos_tweets[index])
        if semantic_bigrams:
            pos_bigrams = get_ngrams(pos_tweets[index], pos_this_tweet,
                                     n=[2], use_just_words=False)
        if lexical:
            words_ngrams = get_ngrams(tweet, tweet_tokens, n=ngram_list, use_just_words=False)
        if sentiment:
            sentiment_features = get_sentiment_features(path, tweet, tweet_tokens, pos_this_tweet)
        features.append({**pragmatic_features, **semantic_features, **pos_bigrams,
                         **words_ngrams, **sentiment_features})
        index += 1
    return features
