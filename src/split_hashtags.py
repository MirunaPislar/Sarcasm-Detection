from nltk.corpus import words
import re, os
import data_processing as data_proc

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
                if wordlist.__contains__(hashtag[j:i].lower()):
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
    word_list = data_proc.load_file(word_file).split()
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
