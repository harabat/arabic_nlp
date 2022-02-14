#!/usr/bin/env python
# coding: utf-8

# # Arabic

# ## Libraries

# In[1]:


# for data analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for English NLP
import spacy

# for Arabic lemmatization and diacritization
from farasa.stemmer import FarasaStemmer
from farasa.diacratizer import FarasaDiacritizer

# for NLP
import nltk

# for string manipulation
import re

# for getting translations
# from reverso_api.context import ReversoContextAPI

# for creating anki decks
# import genanki

# generic
from itertools import islice, permutations
from time import sleep
import pickle


# ## Texts

# In[34]:


# load Arabic text
# with open('/home/warehouse/Nextcloud/Books/Arabic/Fiction/Andy Weir - Artemis - ar.txt', 'r') as file:
#     text_ar = file.read()

with open('./Books/Stephen King, Institute.txt', 'r') as file:
    text_ar = file.read()
    
# text_ar = regex.sub(r"\p{P}+", '  ', text_ar)


# In[3]:


# load English text
with open('/home/warehouse/Nextcloud/Books/Arabic/Fiction/Andy Weir - Artemis - en.txt', 'r') as file:
    text_en = file.read()
    
# text_en = regex.sub(r"\p{P}+", '  ', text_en)


# ## Naive approach

# In[4]:


# naive approach to textual analysis, devoid of any normalisation
def naive_approach(text):
    # tokenization
    words = text.split()
    words_unique = set(text.split())
    
    # count frequencies of unique lemmas
    freq_list = nltk.FreqDist(words)
    
    # determine coverage of total lemmas by the n most frequent lemmas
    coverage = pd.DataFrame({
        'Lemma': dict(freq_list.most_common()).keys(),
        'Values': dict(freq_list.most_common()).values()
    })
    coverage['Cumulative'] = coverage['Values'].cumsum() / coverage['Values'].sum() * 100
    
    elements = {
        'words': words,
        'words_unique': words_unique,
        'freq_list': freq_list,
        'coverage': coverage,
    }
    
    return elements


# ### Arabic text

# In[5]:


# naive approach applied to Arabic text
naive_approach_ar = naive_approach(text_ar)


# In[6]:


# metrics of naive approach
print(
    'words:\t\t', len(naive_approach_ar['words']),
    '\nunique words:\t', len(naive_approach_ar['words_unique'])
)

print(naive_approach_ar['freq_list'].most_common(10))
    
for i in range(0, 101, 10):
    print(i, sum(naive_approach_ar['coverage']['Cumulative'] <= i))

naive_approach_ar['coverage']['Cumulative'].plot()


# ### English text

# In[11]:


# naive approach applied to English text
naive_approach_en = naive_approach(text_en)


# In[12]:


# metrics of naive approach
print(
    'words:\t\t', len(naive_approach_en['words']),
    '\nunique words:\t', len(naive_approach_en['words_unique'])
)

print(naive_approach_en['freq_list'].most_common(10))
    
for i in range(0, 101, 10):
    print(i, sum(naive_approach_en['coverage']['Cumulative'] <= i))

naive_approach_en['coverage']['Cumulative'].plot()


# ## NLP

# Approach:
# - load NLP model
# - process text
# - perform a sanity check
# - lemmatize text
# - identify unique lemmas
# - plot cumulative frequency of unique lemmas
# - perform the above analysis without stopwords
# - check what tokens have been discarded during the analysis

# ### English text

# #### Unique lemmas

# In[13]:


# initiate English NLP model
nlp = spacy.load('en_core_web_md')


# In[14]:


# process text
doc = nlp(text_en)


# In[16]:


# sanity check to estimate the most frequent tokens that fulfill given criteria
def check_freqs(text, keep):
    condition = lambda token, keep: any([
        token.is_alpha,
        token.is_stop,
        (token.is_oov and not token.is_space)
    ]) if keep else all([
        not token.is_alpha,
        not token.is_digit,
        not token.is_punct,
        not token.is_space,
        not token.is_stop
    ])
    
    freqs = nltk.FreqDist([token.lemma_ for token in doc if condition(token, keep)])
    
    return freqs


# In[17]:


# apply sanity check to text
freqs_kept = check_freqs(text_en, True)
freqs_discarded = check_freqs(text_en, False)
print(
    'Kept tokens: {} occurrences from {} tokens\n'.format(
        sum(freqs_kept.values()), len(freqs_kept)
    ),
    freqs_kept.most_common(10),
    '\n\n',
    'Discarded tokens: {} occurrences from {} tokens\n'.format(
        sum(freqs_discarded.values()), len(freqs_discarded)
    ),
    freqs_discarded.most_common(10)
)


# In[18]:


# lemmatization and tokenization
lemmas_en = [
    token.lemma_ for token in doc if (
        token.is_alpha \
        or token.is_stop
        or (token.is_oov and not token.is_space) \
    )
]
len(lemmas_en)


# In[19]:


# identify unique lemmas
lemmas_unique_en = set(lemmas_en)
len(lemmas_unique_en)


# In[23]:


# count frequencies of unique lemmas across kept lemmas
freq_lemmas_en = nltk.FreqDist(lemmas_en)


# In[24]:


# sort in descending order
freq_lemmas_en.most_common(10)


# #### Cumulative frequency

# In[205]:


# determine coverage of total lemmas by the n most frequent lemmas
coverage_en = pd.DataFrame({
    'Lemma': dict(freq_lemmas_en.most_common()).keys(),
    'Values': dict(freq_lemmas_en.most_common()).values(),
})

coverage_en['Cumulative'] = coverage_en['Values'].cumsum() / coverage_en['Values'].sum() * 100


# In[207]:


# plot coverage
coverage_en['Cumulative'].plot()

for i in range(0, 101, 10):
    print(i, sum(coverage_en['Cumulative'] <= i))


# #### Lemmas not considered

# In[71]:


# check discarded tokens to ensure that they are not a large part of the text
def discarded(tokens, lemmas_unique):
#     list conditions for discarding
    conditions = lambda token: all([
        not token.is_alpha,
        not token.is_digit,
        not token.is_punct,
        not token.is_space,
        not token.is_stop
    ])

#     identify unique discarded tokens
    others = [token.lemma_.lower() for token in tokens if conditions(token)]
    others_unique = set(others)
    print(
        'Discarded tokens: {} occurrences from {} tokens\n'.format(
            len(others), len(others_unique)
        )
    )

#     count frequencies of discarded and kept tokens across all tokens
    freq_others = {}
    freq_lemmas = {}
    items = [token.lemma_.lower() for token in tokens]
    
    for other in others_unique:
        freq_others[other] = items.count(other.lower())
    for lemma in lemmas_unique:
        freq_lemmas[lemma] = items.count(lemma.lower())

    print(
        'Total tokens not considered:\t\t{}\nCompared to total tokens considered:\t{:.2%}\n'.format(
            sum(freq_others.values()),
            sum(freq_others.values()) / sum(freq_lemmas.values())
        )
    )

#     sort discarded tokens
    freq_sorted_others = sorted(freq_others, key=freq_others.get, reverse=True)
    print('Most frequent:\n')
    for i in freq_sorted_others:
        print(i, freq_others[i])
        
    return


# In[42]:


# determine the discarded tokens
discarded(doc, lemmas_unique_en)


# #### Stopwords

# In[132]:


# identify unique stopword lemmas
stopwords_en = [token.lemma_ for token in doc if token.is_stop]
stopwords_unique_en = set(stopwords_en)
print(
    'Stopwords: {} occurrences of {} unique stopword lemmas\n'.format(
        len(stopwords_en),
        len(stopwords_unique_en)
    )
)

# determine frequencies of unique stopwords
freq_stops_en = {}
for stop in stopwords_unique_en:
    freq_stops_en[stop] = freq_lemmas_en[stop] / sum(freq_lemmas_en.values())
    
# sort discarded tokens
freq_sorted_stops_en = sorted(freq_stops_en, key=freq_stops_en.get, reverse=True)
print('Most frequent:\n')
for stop in freq_sorted_stops_en[:10]:
    print(stop, '{:.2%}'.format(freq_stops_en[stop]))

# determine cumulative frequencies
coverage_en = pd.DataFrame({
    'Index': freq_stops_en.keys(),
    'Coverage': freq_stops_en.values()
}).sort_values('Coverage', ascending=False, ignore_index=True)
coverage_en['Cumulative'] = coverage_en['Coverage'].cumsum()

# plot cumulative frequencies
coverage_en['Cumulative'].plot()


# In[133]:


# identify unique non-stopword lemmas
# conditions for non-stopwords lead to some overlap however with stopwords
non_stopwords_en = [
    token.lemma_ for token in doc if (
        token.is_alpha \
        and not token.is_stop \
        and token.lemma_ not in stopwords_en
    )
]
non_stopwords_unique_en = set(non_stopwords_en)
print(
    'Non-stopwords: {} occurrences of {} unique non-stopword lemmas\n'.format(
        len(non_stopwords_en),
        len(non_stopwords_unique_en)
    )
)

# determine frequencies of unique non-stopwords
freq_non_stops_en = {}
for non_stop in non_stopwords_unique_en:
    freq_non_stops_en[non_stop] = freq_lemmas_en[non_stop] / sum(freq_lemmas_en.values())

# sort discarded tokens
freq_sorted_non_stops_en = sorted(freq_non_stops_en, key=freq_non_stops_en.get, reverse=True)
print('Most frequent:\n')
for non_stop in freq_sorted_non_stops_en[:10]:
    print(non_stop, '{:.2%}'.format(freq_non_stops_en[non_stop]))

# determine cumulative frequencies
coverage_non_stop_en = pd.DataFrame({
    'Index': freq_non_stops_en.keys(),
    'Coverage': freq_non_stops_en.values()
}).sort_values('Coverage', ascending=False, ignore_index=True)
coverage_non_stop_en['Cumulative'] = coverage_non_stop_en['Coverage'].cumsum()

# plot cumulative frequencies
coverage_non_stop_en['Cumulative'].plot()


# In[160]:


# populate dataframe with lemmas and frequencies
lemmas = pd.DataFrame(freq_lemmas_en.most_common(), columns=['Lemma', 'Freq']).iloc[:2500]
lemmas['Stop'] = lemmas['Lemma'].isin(stopwords_unique_en)
lemmas['Non-stop'] = lemmas['Lemma'].isin(non_stopwords_unique_en)
print(
    'Both stop and non-stop:\t\t{}\nNeither stop nor non-stop:\t{}'.format(
        sum(lemmas['Stop'] & lemmas['Non-stop']),
        sum(~lemmas['Stop'] & ~lemmas['Non-stop'])
    )
)

# plot frequencies of lemmas whether they are stopwords or not
sns.barplot(
    data=lemmas.iloc[:200],
    x=list(range(200)),
    y='Freq',
    hue='Stop',
)

lemmas.head()


# ### Arabic text

# #### Unique lemmas

# In[7]:


# initiate Arabic NLP model
stemmer = FarasaStemmer()


# In[8]:


# lemmatization
text_stemmed_ar = stemmer.stem(text_ar)


# In[9]:


# normalize text by removing diacritics and dealing with different forms of alif
def normalize_ar(token):
    # strip Arabic diacritics
    token = re.compile(r'[\u064B-\u065F]').sub('', token)
    # replace Hamzated Alif with Alif bare
    token = re.compile(r'[\u0622\u0623\u0625]').sub('\u0627', token)
    # replace alifMaqsura with Yaa
    token = re.compile(r'[\u0649]').sub('\u064A', token)
    
    return token


# In[10]:


# normalize text by removing diacritics and dealing with different forms of alif
text_stemmed_ar = normalize_ar(text_stemmed_ar)


# In[11]:


# sanity check to estimate the most frequent tokens that fulfill given criteria
def check_freqs(tokens, keep):
    condition = lambda token, keep: any([
        token.isalpha()
    ]) if keep else all([
        not token.isalpha(),
        not token.isnumeric(),
        not token.isspace(),
        token.isalnum()
    ])
    
    freqs = nltk.FreqDist([token for token in tokens if condition(token, keep)])
    
    return freqs


# In[12]:


# apply sanity check to text
freqs_kept = check_freqs(nltk.wordpunct_tokenize(text_stemmed_ar), True)
freqs_discarded = check_freqs(nltk.wordpunct_tokenize(text_stemmed_ar), False)
print(
    'Kept tokens: {} occurrences from {} tokens\n'.format(
        sum(freqs_kept.values()), len(freqs_kept)
    ),
    freqs_kept.most_common(10),
    '\n\n',
    'Discarded tokens: {} occurrences from {} tokens\n'.format(
        sum(freqs_discarded.values()), len(freqs_discarded)
    ),
    freqs_discarded.most_common(10)
)


# In[13]:


# tokenization of lemmas
lemmas_ar = [token for token in nltk.wordpunct_tokenize(text_stemmed_ar) if token.isalpha()]
len(lemmas_ar)


# In[14]:


# identify unique lemmas
lemmas_unique_ar = set(lemmas_ar)
len(lemmas_unique_ar)


# In[15]:


# count frequencies of unique lemmas across kept lemmas
freq_lemmas_ar = nltk.FreqDist(lemmas_ar)

# pickle frequencies of unique lemmas
with open('freq_lemmas_ar.pkl', 'wb') as file:
    pickle.dump(freq_lemmas_ar, file)


# In[16]:


# sort in descending order
freq_lemmas_ar.most_common(10)


# #### Cumulative frequency

# In[17]:


# determine coverage of total lemmas by the n most frequent lemmas
coverage_ar = pd.DataFrame({
    'Lemma': dict(freq_lemmas_ar.most_common()).keys(),
    'Values': dict(freq_lemmas_ar.most_common()).values(),
})

coverage_ar['Cumulative'] = coverage_ar['Values'].cumsum() / coverage_ar['Values'].sum() * 100


# In[33]:


coverage_ar[120:135]


# In[18]:


# plot coverage
coverage_ar['Cumulative'].plot()

for i in range(0, 101, 10):
    print(i, sum(coverage_ar['Cumulative'] <= i))


# #### Lemmas not considered

# In[19]:


# check discarded tokens to ensure that they are not a large part of the text
def discarded(tokens, lemmas_unique):
#     list conditions for discarding
    conditions = lambda token: all([
        not token.isalpha(),
        not token.isnumeric(),
        not token.isspace(),
        token.isalnum()
    ])

#     identify unique discarded tokens
    others = [token for token in tokens if conditions(token)]
    others_unique = set(others)
    print(
        'Discarded tokens: {} occurrences from {} tokens\n'.format(
            len(others), len(others_unique)
        )
    )

#     count frequencies of discarded and kept tokens across all tokens
    freq_others = {}
    freq_lemmas = {}

    for other in others_unique:
        freq_others[other] = tokens.count(other)
    for lemma in lemmas_unique:
        freq_lemmas[lemma] = tokens.count(lemma)

    print(
        'Total tokens not considered:\t\t{}\nCompared to total tokens considered:\t{:.2%}\n'.format(
            sum(freq_others.values()),
            sum(freq_others.values()) / sum(freq_lemmas.values())
        )
    )
    
#     sort discarded tokens
    freq_sorted_others = sorted(freq_others, key=freq_others.get, reverse=True)
    print('Most frequent:\n')
    for i in freq_sorted_others:
        print(i, freq_others[i])
    
    return


# In[20]:


# determine the discarded tokens
discarded(nltk.wordpunct_tokenize(text_stemmed_ar), lemmas_unique_ar)


# #### Stopwords

# In[22]:


# load list of Arabic stopwords
with open('stopwords.txt', 'r') as file:
    stopwords_all_ar = file.read()
    
# normalise the stopwords through dediacritization
stopwords_all_ar = normalize_ar(stopwords_all_ar)


# In[23]:


# identify unique stopword lemmas
stopwords_ar = [token for token in stopwords_all_ar.split() if token.isalpha()]
stopwords_unique_ar = set(stopwords_ar)
print(
    'Stopwords: {} occurrences of {} unique stopword lemmas\n'.format(
        len(stopwords_ar),
        len(stopwords_unique_ar)
    )
)

# determine frequencies
freq_stops_ar = {}
for stop in stopwords_unique_ar:
    try:
        freq_stops_ar[stop] = freq_lemmas_ar[stop] / sum(freq_lemmas_ar.values())
    except:
        print(stop)
        continue
        
# sort discarded tokens
freq_sorted_stops_ar = sorted(freq_stops_ar, key=freq_stops_ar.get, reverse=True)
print('Most frequent:\n')
for stop in freq_sorted_stops_ar[:10]:
    print(stop, '\n\t{:.2%}'.format(freq_stops_ar[stop]))
    
# determine cumulative frequencies
coverage_ar = pd.DataFrame({
    'Index': freq_stops_ar.keys(),
    'Coverage': freq_stops_ar.values()
}).sort_values('Coverage', ascending=False, ignore_index=True)
coverage_ar['Cumulative'] = coverage_ar['Coverage'].cumsum()

# plot cumulative frequencies
coverage_ar['Cumulative'].plot()


# In[24]:


# identify unique non-stopword lemmas
# conditions for non-stopwords lead to some overlap however with stopwords
non_stopwords_ar = [
    token for token in nltk.wordpunct_tokenize(text_stemmed_ar) if (
        token.isalpha() \
        and token not in stopwords_unique_ar
    )
]
non_stopwords_unique_ar = set(non_stopwords_ar)
print(
    'Non-stopwords: {} occurrences of {} unique non-stopword lemmas\n'.format(
        len(non_stopwords_ar),
        len(non_stopwords_unique_ar)
    )
)

# determine frequencies
freq_non_stops_ar = {}
for non_stop in non_stopwords_unique_ar:
    freq_non_stops_ar[non_stop] = freq_lemmas_ar[non_stop] / sum(freq_lemmas_ar.values())
    
# sort discarded tokens
freq_sorted_non_stops_ar = sorted(freq_non_stops_ar, key=freq_non_stops_ar.get, reverse=True)
print('Most frequent:\n')
for non_stop in freq_sorted_non_stops_ar[:10]:
    print(non_stop, '{:.2%}'.format(freq_non_stops_ar[non_stop]))

# determine cumulative frequency
coverage_non_stop_ar = pd.DataFrame({
    'Index': freq_non_stops_ar.keys(),
    'Coverage': freq_non_stops_ar.values()
}).sort_values('Coverage', ascending=False, ignore_index=True)
coverage_non_stop_ar['Cumulative'] = coverage_non_stop_ar['Coverage'].cumsum()

# plot cumulative frequencies
coverage_non_stop_ar['Cumulative'].plot()


# In[25]:


# populate dataframe with lemmas and frequencies
lemmas = pd.DataFrame(freq_lemmas_ar.most_common(), columns=['Lemma', 'Freq']).iloc[:2500]
lemmas['Stop'] = lemmas['Lemma'].isin(stopwords_unique_ar)
lemmas['Non-stop'] = lemmas['Lemma'].isin(non_stopwords_unique_ar)
print(
    'Both stop and non-stop:\t\t{}\nNeither stop nor non-stop:\t{}'.format(
        sum(lemmas['Stop'] & lemmas['Non-stop']),
        sum(~lemmas['Stop'] & ~lemmas['Non-stop'])
    )
)

# plot frequencies of lemmas whether they are stopwords or not
sns.barplot(
    data=lemmas.iloc[:200],
    x=list(range(200)),
    y='Freq',
    hue='Stop',
)

lemmas.head()


# ## Online sentences and translations

# ### Frequency list and functions

# In[14]:


# load Arabic frequency list from text
with open('freq_lemmas_ar.pkl', 'rb') as file:
    freq_lemmas_ar = pickle.load(file)


# In[15]:


# normalize text by removing diacritics and dealing with different forms of alif
def normalize_ar(token):
    # strip Arabic diacritics
    token = re.compile(r'[\u064B-\u065F]').sub('', token)
    # replace Hamzated Alif with Alif bare
    token = re.compile(r'[\u0622\u0623\u0625]').sub('\u0627', token)
    # replace alifMaqsura with Yaa
    token = re.compile(r'[\u0649]').sub('\u064A', token)
    
    return token


# In[9]:


# keep only the first token from every row of a frequency list that is not from text
def normalize_freq_list(freq_list):
    keys = freq_list        .apply(normalize_ar)        .apply(lambda x: re.sub(r'^\W+', '', x))        .apply(lambda x: re.sub(r'\W+.*', '', x))        .drop_duplicates()        .reset_index(drop=True)
    
    freq_list = {
        key: len(keys) - index for index, key in keys.iteritems()
    }
    
    return freq_list


# In[250]:


# load Arabic frequency list (not from text)
freq_list_full_ar = pd.read_csv('Anki decks/freqlist.csv', header=0)
    
freq_list_ar = normalize_freq_list(freq_list_full_ar['Arabic'])

freq_list_ar


# ### Reverso

# #### API functions

# In[326]:


# get translations and example sentences from reverso
def reverso(word):
    try:
        api = ReversoContextAPI(
            source_text=word,
            source_lang='ar',
            target_lang='en'
        )

        translations = []
        for i in islice(api.get_translations(), 10):
#             try:
            translations.append(i)
#             except Exception as ex:
#                 print(
#                     'error\t\t\t\t\t\t\treverso translations:\t',
#                     repr(ex)
#                 )
#                 sleep(10)
#                 translations.append(i) 

        examples = []
        for i in islice(api.get_examples(), 10):
#             try:
            examples.append(i)
#             except Exception as ex:
#                 print(
#                     'error\t\t\t\t\t\t\treverso examples:\t',
#                     repr(ex)
#                 )
#                 sleep(10)
#                 examples.append(i)

        result = {
            'translations': translations,
            'examples': examples
        }
    
    except Exception as ex:
        print(
            'error\t\t\t\t\t\t\treverso:\t',
            repr(ex)
        )
        sleep(10)
        result = reverso(word)
    
    return result


# In[327]:


# iterate over lemmas and deal with connection errors
def api_calls(freq_list_translations, freq_list, n):
    try:
        keys = list(freq_list.keys())[n:]
        for word in keys:
#             try:
            freq_list_translations[word] = reverso(word)
#             except Exception as ex:
#                 print(
#                     'error\t\t\t\t\t\t\tapi_calls for:\t',
#                     repr(ex)
#                 )
#                 sleep(10)
#                 freq_list_translations[word] = reverso(word)
            print(n, word)
            n += 1
    except Exception as ex:
        print(
            'error\t\t\t\t\t\t\tapi_calls:\t',
            repr(ex)
        )
        sleep(10)
        freq_list_translations = api_calls(freq_list_translations, n)
    
    return freq_list_translations


# #### Getting translations and example sentences from Reverso

# In[379]:


# list of lemmas to get sentences for
freq_list = dict(freq_lemmas_ar.most_common(2500))
print(len(freq_list))

# remove any lemmas that already have translations
for i in set(freq_list.keys()).intersection(set(freq_list_translations.keys())):
    freq_list.pop(i)
print(len(freq_list))


# In[315]:


# get translations and examples for each lemma in freq_list
freq_list_translations = api_calls(freq_list_translations, freq_list, 0)


# In[381]:


# check that freq_list_translations has covered unique lemmas
print(
    'freq_list: {}\nfreq_list_translations: {}\nEqual: {}\nDifference: {} {}'.format(
        len(dict(freq_lemmas_ar.most_common(2500))),
        len(freq_list_translations),
        set(freq_list_translations.keys()) == set(dict(freq_lemmas_ar.most_common(2500)).keys()),
        len(set(dict(freq_lemmas_ar.most_common(2500)).keys()).difference(set(freq_list_translations.keys()))),
        len(set(freq_list_translations.keys()).difference(set(dict(freq_lemmas_ar.most_common(2500)).keys())))
    )
)


# In[408]:


# pickle reverso translations
with open('translations_reverso.pkl', 'wb') as file:
    pickle.dump(freq_list_translations, file)


# #### Scoring sentences

# In[3]:


# load reverso translations
with open('translations_reverso.pkl', 'rb') as file:
    freq_list_translations = pickle.load(file)


# In[57]:


# check the sentences
sentences = []
translations = []

for key, value in freq_list_translations.items():
    for item in value['examples']:
        sentences.append(item[0].text)
        translations.append(item[1].text)
        
sentences = pd.DataFrame({
    'text_source': sentences,
    'text_target': translations
})

sentences.drop_duplicates(
    'text_source',
    keep='first',
    inplace=True,
    ignore_index=True
)

sentences.drop_duplicates(
    'text_target',
    keep='first',
    inplace=True,
    ignore_index=True
)

sentences.sort_values('text_source')


# In[25]:


# initiate Arabic NLP model
# stemmer in interactive mode to increase speed when iterating
stemmer = FarasaStemmer(interactive=True)


# In[26]:


# function that scores sentences based on how common their lemmas are
def score_sentence(sentence, freq_lemmas_ar):
    score = 0
    lemmas = 0
    
    sentence_normalized = normalize_ar(stemmer.stem(sentence))
    sentence_lemmas = [token for token in nltk.wordpunct_tokenize(sentence_normalized) if token.isalpha()]
    
    for lemma in sentence_lemmas:
        try:
            score += freq_lemmas_ar[lemma]
            lemmas += 1
        except:
            continue
            
    score = score / lemmas / len(sentence_lemmas) * 100
            
    return score


# In[27]:


# check the scoring function
word = list(dict(freq_lemmas_ar.most_common(2500)).keys())[10]
print(word)

sentence = freq_list_translations[word]['examples']
for ar, en in sentence:
    print(
        ar.text,
        '\n\t\t\t',
        score_sentence(ar.text, freq_lemmas_ar),
        '\n\t\t\t',
        en.text,
        '\n'
    )
   


# In[28]:


# iterate over sentences, scoring and translating each
def score_translate_sentences(freq_list_translations, freq_lemmas_ar):
    sentences = {}

    for lemma in freq_list_translations:
        examples = freq_list_translations[lemma]['examples']
        for sentence_ar, sentence_en in examples:
            sentences[sentence_ar.text] = {
                'translation': sentence_en.text,
                'score': score_sentence(sentence_ar.text, freq_lemmas_ar)
            }

    return sentences


# In[29]:


# score sentences
sentences = score_translate_sentences(freq_list_translations, freq_lemmas_ar)


# In[30]:


# check best scoring sentences
sentences_sorted = sorted(sentences, key=lambda x: sentences[x]['score'], reverse=True)
for i in sentences_sorted[:10]:
    print(
        '{}\n{}\n'.format(
        i,
        sentences[i]
        )
    )


# In[31]:


# pickle reverso sentences
with open('sentences_reverso.pkl', 'wb') as file:
    pickle.dump(sentences, file)


# ### Tatoeba

# #### Getting sentences from Tatoeba

# In[87]:


# load the dataset from Tatoeba's downloads page
sentences_all = pd.read_csv(
    'Sentences/sentences.csv',
    delimiter='\t',
    header=None,
    names=['id', 'lang', 'text']
)

links = pd.read_csv(
    'Sentences/links.csv',
    delimiter='\t',
    header=None,
    names=['id', 'translation_id']
)

reviews = pd.read_csv(
    'Sentences/users_sentences.csv',
    delimiter='\t',
    header=None,
    names=[
        'username',
        'id',
        'review',
        'added',
        'modified'
    ]
)

# check dataframes' shape
print(
    'Shapes:',
    '\nsentences', sentences_all.shape,
    '\nlinks', links.shape,
    '\nreviews', reviews.shape
)


# In[88]:


# select sentences in either eng or ara
sentences_ara_eng = sentences_all[sentences_all['lang'].isin(['ara', 'eng'])]

print(
    'Shape:', sentences_ara_eng.shape,
    '\n\nValue counts:\n',
    sentences_ara_eng['lang'].value_counts()
)

sentences_ara_eng.head()


# In[254]:


# ensure that 'links' dataframe is bidirectional
print(
    all(links['id'].value_counts() == links['translation_id'].value_counts())
)

links.head()


# In[89]:


# determine average scores
reviews_id = reviews.groupby('id', as_index=False).agg({
    'review': 'mean'
})

reviews_id[reviews_id['review'] < 0]


# In[90]:


# select inter-related sentences that are either 'ara' or 'eng', and add scores
sentences_translations = links.merge(
    sentences_ara_eng,
    how='inner',
    on='id'
) \
.merge(
    sentences_ara_eng,
    how='inner',
    left_on='translation_id',
    right_on='id',
    suffixes=('_source', '_target')
) \
.drop(columns=['translation_id'])

sentences_translations


# In[91]:


# add review information for source and target
sentences_scores = sentences_translations.merge(
    reviews_id,
    how='left',
    left_on='id_source',
    right_on='id'
) \
.merge(
    reviews_id,
    how='left',
    left_on='id_target',
    right_on='id',
    suffixes=('_score_source', '_score_target')
) \
.drop(columns=['id_score_source', 'id_score_target']) \
.fillna(0)

sentences_scores


# In[92]:


# select only sentences are 'ara' to 'eng' and have positive scores, then remove 'ara' duplicates
# 'eng' duplicates will be dropped in the next section
sentences_ara = sentences_scores[
    (
        (sentences_scores['lang_source'] == 'ara') 
        & (sentences_scores['lang_target'] == 'eng')
    ) & (
        (sentences_scores['review_score_source'] >= 0)
        & (sentences_scores['review_score_target'] > 0)
    )
].sort_values(
    ['review_score_source', 'review_score_target'],
    ascending=False
).drop_duplicates(
    ['id_source'],
    keep='first'
).sort_values(
    'id_target',
    ignore_index=True
)

sentences_ara


# #### Removing duplicates

# In[93]:


# initiate Arabic NLP model
# stemmer in interactive mode to increase speed when iterating
stemmer = FarasaStemmer(interactive=True)


# In[94]:


# identify sentences with identical translations
duplicates = sentences_ara[sentences_ara.duplicated('id_target', keep=False)]

print(len(duplicates))
duplicates.head()


# In[95]:


# lemmatize, normalize, and tokenize sentences for comparison

for index, row in duplicates.iterrows():
    sentence_normalized = normalize_ar(stemmer.stem(row['text_source']))
    sentence_lemmas = [token for token in nltk.wordpunct_tokenize(sentence_normalized) if token.isalpha()]
    duplicates.loc[index, 'lemmas_source'] = {'_': sentence_lemmas}.values()
        
print(len(duplicates))
duplicates.head()


# In[98]:


# check overlap between lemmas of duplicate sentences
redundant_all = set()

for i in duplicates['id_target'].unique():
    duplicate_sentences = duplicates[duplicates['id_target'] == i]
    redundant = set()

#     if lemmas of sentence are included in lemmas of another, get the index
    for j in permutations(duplicate_sentences.index, 2):
        difference = set(duplicates.loc[j[0], 'lemmas_source']).difference(
            set(duplicates.loc[j[1], 'lemmas_source'])
        )
        if not difference:
            redundant.add(j[0])
    
#     if all sentences for specific translation are redundant, remove one from 'redundant'
    if set(duplicate_sentences.index) == redundant:
        redundant.pop()
        
    redundant_all = redundant_all.union(redundant)
            
len(set(redundant_all))


# In[99]:


# check redundant duplicate sentences
duplicates[duplicates.index.isin(redundant_all)]


# In[100]:


# drop redundant duplicate sentences
index_drop = duplicates[duplicates.index.isin(redundant_all)].index
sentences_ara.drop(index=index_drop, inplace=True)

sentences_ara


# In[101]:


# pickle freq_list_translations
with open('translations_tatoeba.pkl', 'wb') as file:
    pickle.dump(sentences_ara, file)


# #### Scoring sentences

# In[23]:


# load tatoeba translations
with open('translations_tatoeba.pkl', 'rb') as file:
    sentences_ara = pickle.load(file)


# In[3]:


# initiate Arabic NLP model
# stemmer in interactive mode to increase speed when iterating
stemmer = FarasaStemmer(interactive=True)


# In[28]:


# function that scores sentences based on how common their lemmas are
def score_sentence(sentence, freq_lemmas_ar):
    score = 0
    lemmas = 0
    
    sentence_normalized = normalize_ar(stemmer.stem(sentence))
    sentence_lemmas = [token for token in nltk.wordpunct_tokenize(sentence_normalized) if token.isalpha()]
    
    for lemma in sentence_lemmas:
        try:
            score += freq_lemmas_ar[lemma]
            lemmas += 1
        except:
            continue
            
    score = score / lemmas / len(sentence_lemmas) * 100
            
    return score


# In[29]:


# check the scoring function
sentence = sentences_ara.iloc[30]
print(sentence)

score_sentence(sentence['text_source'], freq_lemmas_ar)


# In[30]:


# iterate over sentences, scoring and translating each
def score_translate_sentences(sentences_ara, freq_lemmas_ar):
    sentences = {}

    for index, row in sentences_ara.iterrows():
        sentences[row['text_source']] = {
            'translation': row['text_target'],
            'score': score_sentence(row['text_source'], freq_lemmas_ar)
        }
    
    return sentences


# In[19]:


# score sentences
sentences = score_translate_sentences(sentences_ara, freq_lemmas_ar)


# In[108]:


# load diactritization model
diacritizer = FarasaDiacritizer(interactive=True)


# In[109]:


# diacritize sentences
sentences = {diacritizer.diacritize(key): value for key, value in sentences.items()}


# In[110]:


# check correlation between length of sentences and score
# longer sentences have higher scores, because of how score is calculated
sentences_ara['score_source'] = pd.NA

for index, text in sentences_ara['text_source'].iteritems():
    sentences_ara.loc[index, 'score_source'] = sentences[text]['score']
    
x = sentences_ara[
    sentences_ara.duplicated(
        'id_target',
        keep=False
    )
].copy()

x['len'] = x['text_source'].apply(len)

x.sort_values(
    ['id_target', 'len', 'score_source'],
    ascending=False,
    inplace=True
)
x = x.where(
    x['score_source']/x['score_source'].max() >= 0
).dropna()

fig = plt.figure(figsize=(10, 10))

x1 = x.drop_duplicates('id_target', keep='first')

ax1 = fig.add_subplot(221)
sns.histplot(
    data=x1['len']/x['len'].max(),
    color='red',
    label='len',
    ax=ax1
)
sns.histplot(
    x1['score_source']/x['score_source'].max(),
    color='blue',
    label='score',
    ax=ax1
)
ax1.set_title('keep longest sentences')
ax1.legend()

x2 = x.drop_duplicates('id_target', keep='last')

ax2 = fig.add_subplot(222)
sns.histplot(
    data=x2['len']/x['len'].max(),
    color='red',
    label='len',
    ax=ax2
)
sns.histplot(
    x2['score_source']/x['score_source'].max(),
    color='blue',
    label='score',
    ax=ax2
)
ax2.set_title('keep shortest sentences')
ax2.legend()

ax3 = fig.add_subplot(223)
sns.scatterplot(
    data=x,
    x='len',
    y='score_source',
    ax=ax3
)


# In[20]:


# check best scoring sentences
sentences_sorted = sorted(sentences, key=lambda x: sentences[x]['score'], reverse=True)
for i in sentences_sorted[:10]:
    print(
        '{}\n{}\n'.format(
        i,
        sentences[i]
        )
    )


# In[112]:


# pickle tatoeba sentences
with open('sentences_tatoeba.pkl', 'wb') as file:
    pickle.dump(sentences, file)


# ## Anki deck generation

# ### Sentences

# In[114]:


# load reverso sentences
with open('sentences_reverso.pkl', 'rb') as file:
    sentences_reverso = pickle.load(file)
    
# for i in range(len(sentences_reverso) - 20):
#     sentences_reverso.popitem()
print(len(sentences_reverso))
# sentences_reverso


# In[115]:


# load tatoeba sentences
with open('sentences_tatoeba.pkl', 'rb') as file:
    sentences_tatoeba = pickle.load(file)
    
# for i in range(len(sentences_tatoeba) - 20):
#     sentences_tatoeba.popitem()
print(len(sentences_tatoeba))
# sentences_tatoeba


# In[121]:


fig, ax = plt.subplots()

sns.histplot(
    [sentences_tatoeba[i]['score'] for i in sentences_tatoeba if sentences_tatoeba[i]['score'] < 50000],
    color='red',
    label='tatoeba',
    alpha=.5,
    ax=ax
)

sns.histplot(
    [sentences_reverso[i]['score'] for i in sentences_reverso if sentences_reverso[i]['score'] < 50000],
    color='blue',
    label='reverso',
    alpha=.5,
    ax=ax
)

ax.set_title('Sentence score counts')
ax.legend()


# ### Card model

# In[122]:


# create random model id
np.random.randint(1<<30, 1<<31)


# In[123]:


arabic_format = '''
<div style="padding: 5%; background-color: lightgray; color: black">
    <div id=arabic style="text-align: center; vertical-align: middle; direction: rtl; color: DarkRed"> 
        {{Arabic}} 
    </div>
    <br>
    {{#Synonyms}}
        (&ne; {{Synonyms}})
    {{/Synonyms}}
</div>

{{hint::Mnemonic}}
'''

english_format = '''
<div style="padding: 5%; background-color: lightgray; color: black">
    <div id=arabic style="text-align: center; vertical-align: middle; direction: rtl; color: DarkRed"> 
        {{Arabic}} 
    </div>
    <br>
    {{#Synonyms}}
        (&ne; {{Synonyms}})
    {{/Synonyms}}
</div>

<hr id=answer>

<div  id=english style='padding:5%; vertical-align: top; background-color:lightgreen; color:black'> 
    {{English}}
</div>

<hr>

<div style="padding:5%;font-size: small; font-weight: regular; direction: ltr;background-color:lightgreen;color:black;">
    <div id='lemmas' style="text-align: justify ; font-weight: regular; direction: rtl"> 
        {{Lemmas}}
    </div>
</div>

<hr>

<div  style="padding-right:10%;padding-left:10%;text-align: justify ; font-size: small; font-weight: regular; direction: ltr;"> 
    | Score: {{Score}} <br> 
    | Mnemonic: {{Mnemonic}}
</div>
'''

css = '''
.card {
    font-family: Arial;
    font-size: x-large;
    text-align: center;
}
#arabic {
    font-family: Noto Sans Arabic UI Lt;
    font-size: xxx-large
}
#english {
    font-family: Noto Sans;
    font-size: xx-large;
    text-align: center;
    vertical-align: middle;
    color: black;
}
#lemmas {
    font-family: Noto Sans Arabic UI Lt;
    font-size: xx-large;
}
'''


# In[124]:


# create deck model
sentence_model = genanki.Model(
    model_id=1886263227,
    name='Arabic',
    fields=[
        {'name': 'Score'},
        {'name': 'Arabic', 'rtl': True},
        {'name': 'English'},
        {'name': 'Mnemonic'},
        {'name': 'Synonyms'},
        {'name': 'Lemmas', 'rtl': True}
    ],
    templates=[
        {
            'score': '{{Score}}',
            'name': 'Card',
            'qfmt': arabic_format,
            'afmt': english_format
        }
    ],
    css=css
)


# ### Reverso

# In[42]:


# create reverso deck
deck_reverso = genanki.Deck(
    1177887818,
    'Arabic: Reverso'
)


# In[43]:


# build deck
for sentence in sentences_reverso:
    translation = sentences_reverso[sentence]['translation']
    score = int(sentences_reverso[sentence]['score'])
#     if score < 200:
#         continue
    
    card = genanki.Note(
        model=sentence_model,
        fields=[
            str(score),
            sentence,
            translation,
            '',
            '',
            ''
        ]
    )
    deck_reverso.add_note(card)


# In[44]:


# save deck
genanki.Package(deck_reverso).write_to_file('deck_reverso.apkg')


# ### Tatoeba

# In[125]:


# create tatoeba deck
deck_tatoeba = genanki.Deck(
    1667002487,
    'Arabic: Tatoeba'
)


# In[126]:


# build deck
for sentence in sentences_tatoeba:
    translation = sentences_tatoeba[sentence]['translation']
    score = int(sentences_tatoeba[sentence]['score'])
#     if score < 200:
#         continue
    
    card = genanki.Note(
        model=sentence_model,
        fields=[
            str(score),
            sentence,
            translation,
            '',
            '',
            ''
        ]
    )
    deck_tatoeba.add_note(card)


# In[127]:


# save deck
genanki.Package(deck_tatoeba).write_to_file('deck_tatoeba.apkg')


# In[ ]:




