# Resources

The resources provided here are only partial. The main reason why these samples have been attached is just to get you started quickly but, for a thorough analysis, the whole */res* directory should be replaced with the full version available for download (or simply for visualization) at [this link](https://drive.google.com/open?id=1AcGulyTXcrsn6hStefD3M0MNrzkxV_1n). 

Below are some descriptions of each group of resources that I used in my analysis and comparisons.

### Datasets

Each dataset directory has a minimum of 8 files (all having the same names) comprising of the tokens, part of speech tags and labels for both the train and test sets as well as the original data files.

* **ghosh:**  this is the project's main dataset, collected by Aniruddha Ghosh and Tony Veale also available on their [repository](https://github.com/AniSkywalker/SarcasmDetection); this directory contains additional files for quick experiments (*train_sample* and *test_sample*) with their afferent tokens, pos tags and labels.

* **sarcasmdetection:** collected by Mathieu Cliche and named in this rather vague way because this dataset was used to build *thesarcasmdetector* website, accessible [here](http://www.thesarcasmdetector.com/about/); other papers refer to it under the same name.

* **riloff:** collected by Ellen Riloff, available for download on her [publications' page](http://www.cs.utah.edu/~riloff/publications_chron.html); this is also the project's oldest and smallest dataset so the tweets are a bit outdated and a lot of them have been removed since 2013 when the tweet IDs were initially collected.

* **hercig:** collected by Tomáš Hercig (previously named Ptáček) and Ivan Habernal, obtained
from their resources [page](http://liks.fav.zcu.cz/sarcasm/) under their consent. This is also the dataset on which the current state-of-the-art has been achieved.

* **demo:** just a subset of Ghosh's dataset that I used for the demo.


<table style="width:200%" align="center">
  <tr>
    <th rowspan="2">Corpus</th>
    <th colspan="2">Train Set</th> 
    <th colspan="2">Test Set</th>
  </tr>
  <tr>
    <th>Sarcastic</th>
    <th>Non-sarcastic</th>
    <th>Sarcastic</th>
    <th>Non-sarcastic</th>
  </tr>
  <tr>
    <td>Ghosh</td>
    <td>24,453</td> 
    <td>26,736</td>
    <td>1,419</td>
    <td>2,323</td>
  </tr>
  <tr>
    <td>Riloff</td>
    <td>215</td> 
    <td>1,153</td>
    <td>93</td>
    <td>495</td>
  </tr>
  <tr>
    <td>SarcasmDetector</td>
    <td>10,000</td> 
    <td>10,000</td>
    <td>2,000</td>
    <td>2,000</td>
  </tr>
  <tr>
    <td>Ptacek</td>
    <td>9,200</td> 
    <td>5,140</td>
    <td>2,300</td>
    <td>1,285</td>
  </tr>
</table>

**Note:** For some of the above datasets, only the tweet IDs have been made publicly available and not the actual text bodies. Therefore, a considerable amount of them has been deleted since the dataset creation, leaving us with subsets of the originals.

**Important:** these datasets are uploaded for convenience purposes only. I do not claim any rights on them so you should use them at your own responsibility. Make sure that you respect their licence and cite the original papers and the authors who so kindly made them available to the research community.

### DeepMoji

This directory is solely based on MIT's deepmoji [project](https://deepmoji.mit.edu/). I adapted the code on their [repo](https://github.com/MirunaPislar/DeepMoji) to collect some dataframe *.csv* files for Ghosh's train and test sets. Each row of the dataframe contains the following information:

* the text of the actual tweet
* overall confidence in the prediction made (a number between 0 and 1)
* indices for the top 5 predicted deepmojis (according to the [emoji/wanted_emojis.txt](emoji/wanted_emojis.txt))
* a confidence number for each of the 5 predicted deepmoji (how suitable is a predicted deepmoji for the current tweet)

### Emoji

* [emoji_frequencies.txt](emoji/emoji_frequencies.txt) contains the most popular emojis, sorted by their occurence number
* [emoji_list.txt](emoji/emoji_list.txt) contains all the emojis and their description
* [emoji_negative_samples.txt](emoji/emoji_negative_samples.txt) and [emoji_positive_samples.txt](emoji/emoji_positive_samples.txt) contain the samples used to train the *emoji2vec* model
* [emoji_sentiment_raw.txt](emoji/emoji_sentiment_raw.txt) and  [emoji_sentiment_dictionary.txt](emoji/emoji_sentiment_dictionary.txt) contain the underlying sentiments' probabilities associated with each emoji (positive, negative and neutral calculated based on context occurrences) 
* [wanted_emojis.txt](emoji/wanted_emojis.txt) contains the 64 deepmojis mapped to their index as proposed by Bjarke Felbo et al. in their [paper](https://arxiv.org/pdf/1708.00524.pdf) (original depiction [here](https://github.com/MirunaPislar/DeepMoji/blob/master/emoji_overview.png))

### GloVe

Contains the global vector representations for words gathered from the original GloVe [page](https://nlp.stanford.edu/projects/glove/).

Alternatively, one can download them directly:

```{r, engine='bash', count_lines}
! wget -q http://nlp.stanford.edu/data/glove.6B.zip
! unzip -q -o glove.6B.zip
```

### Tokens and POS

Two directories obtained by collecting separately the tokens and the part of speech tags of the train/test data (for Ghosh's dataset which is also the default dataset) resulted after applying the [CMU Tweet POS Tagger](http://www.cs.cmu.edu/~ark/TweetNLP/#pos). Several experiments have been conducted on the data to draw some useful conclusions on the particularities of Twitter sarcasm:

* **original_train.txt** - Ghosh's original dataset, no pre-processing applied

* **clean_original_train.txt** - on the *original_train.txt* perform:
    * split around special characters
    * all #sarca* are removed
    * URLs are removed
    * all user mentions are replaced with *@user*
    * hashtags are split and a # sign is appended to every word in the split

**Note** that any #sarcasm tags obtained after the hashtag splitting process should not be removed.

![Pipeline to gather the clean tokens](../images/pipeline_clean_tokens.png)

* **filtered_clean_original_train.txt** - on the *clean_original_train.txt* perform:
    * lower-case and lemmatize every word
    * stopwords are removed

* **grammatical_train.txt** - on the *clean_original_train.txt* perform:
    * the # sign before every hashtag is removed (not the hashtag itself)
    * case is preserved, but words are lemmatized
    * all hyphens at the beginning of a word are removed
    * all contracted forms are expanded
    * all repeating characters are removed and checked against a dictionary
    * emojis are left as they are
    * emoticons are translated to emojis
    
![Pipeline to gather the grammatical tokens](../images/pipeline_gramm_tokens.png)

* **finest_grammatical.txt** - on the *clean_original_train.txt* perform:
    * the sign # before each hashtag is removed (not the hashtag itself)
    * everything is lower-cased, words are lemmatized
    * all hyphens at the beginning of a word are removed
    * all contracted forms are expanded
    * all repeating characters are removed and checked against a dictionary
    * emojis are translated to their descriptions
    * emoticons are translated to their descriptions
    * slang is corrected, abbreviations are expanded

* **strict_train.txt** - the original dataset is cleared completely of hashtags, emojis, URLs and user mentions. 

**Note** that some of the lines might be empty after this clearing process.

### Topic Data

Contains the corpora and dictionaries used to train multiple LDA models in the topic analysis phase (i.e they have various numbers of topics/passes and different degrees of restrictiveness on the words allowed for topic training).

### Vocabulary

Contains multiple vocabularies built on Ghosh's training data with various degrees of filtering to accomplish different tasks.

### Other useful files

* the [MPQA subjectivity lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/) used to extract sentiment features
* [word_list.txt](word_list.txt) and [word_list_freq.txt](word_list_freq.txt) are kind of *dictionaries* used to better split hashtags
* [stopwords.txt](stopwords.txt) and [stopwords_loose.txt](stopwords_loose.txt) are two lists of commonly occurring words that are used to achieve different levels of filtering over the corpus

