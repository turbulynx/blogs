---
title: Generative AI - NLP 
date: 2024-07-11
tags: ["Chatbots", "RASA", "ChatGPT", "BERT", "Transformers", "Prompt Engineering"]
image : "/img/posts/natural-language-processing.jpeg"
Description  : "Generative AI with NLP LLM: 
"
---
# Objective

# 1. Introduction
---
## NLP Tasks
- **Language Modelling** : Predict the next word based on the sequence of words that already occurs in a given language. Application: speech recognition, OCR Translation etc.
- **Text classification** : Assigning a text into one of the known categories based on content. Application: Email spam, sentiment analysis etc
- **Information Extraction** : Extracting important information from a text. Application: extracting user's intent from input text, calendar etc.
- **Information Retrieval** : Finding data based on user query. Application: Used in search engine.
- **Conversational Agent** : A Dialogue system that can converse in a human language. Application: Siri, Alexa etc.
- **Text Summarization** : Short summary of longer documents by retaining the important information. Application: Summary report generation from social media information.
- **Question Answering** : Automatically answer questions posted in Natural Language. Application: Answering a user query based on data from a database.
- **Machine Translation** : Converting a piece of text from one to another language. Application: Google transalator
- **Topic Modelling** : Uncover the topical structure of large collection of text. Application: Text Mining

## Understanding Human language and its building blocks
- **Language**: words used in a Structured and conventional way and used to convey an idea by speech, writing or gesture.
- **Linguistics**: Scientific study of a language and its structure. Study of language grammer, syntax and phonitics.
  - Building Blocks:
    - Phonemes: smallest unit of speech & sound. English language has 44 of them. Applications: Speech to text transcriptions and text to speech conversations.
    - morphemes and lexemes: Applications: Tokenization, Stemming, lemmatization, word embedding, parts of speech tagging.
       - morphemes: smallest unit of a word. not all morphemes are words but the prefixes and suffixes are. e.g. 'multi' in multistory.
       - lexemes: basic building block of a language. dictionary entries are lexems. lexemes are built on basic form e.g. walk, walking, walked.
 - **Syntax**: arragnement of words in a sentence. Representation of sentence is done using parse tree. Entity Extraction and relation extraction.
    - syntax - phrases and sentences
    - context - meaning
    - Syntax Parse Tree:
      ![](/blogs/img/posts/syntax-parse-tree.png)
      - NP - noun phrase
      - VP - verb phrase
      - PP - prepositional phrase
      - S - sentence at the highest level.
- **Context**: words and sentences that surround any part of discourse and that helps determine the meaning. Application: Sarcasm detection, summarization, topic modelling. Made up of:
  - semantics: direct meaning 
  - pragmatics: adds world knowledge and external knowledge.

## Challenges of NLP
- Ambiguity: two or more meanings of a single passage. e.g. we saw her duck. Common knowledge assumptions. e.g he says Sun rises in the west (assumption that a preson knows sun rises in the east)
- Creativity

---
# 2. Pipeline of NLP
---
## NLP Pipeline
Step by step processing of text is known as NLP Pipeline:
- Data collection (scrapy)
- Text Cleaning 
- Pre-processing (stemming and lemmetization)
- Feature engineering (one hot encoding, bag of words technique)
- Modeling
- Evaluation
- Deployment
- Monitoring
---

---
## NLTK library
NLTK library is most commonly used NLP library. Common text pre-processing steps in NLP: 
  - Tokenization: breaking up text into smaller pieces called tokens. 
  - Stemming
  - Lemmatization
  - Word Embedding
  - Parts of speech tagging
  - Stop Word removal
  - Word Sence disambiguation
  - Named Entity Recognition (NER)
    
### Tokenization: breaking up text into smaller pieces called tokens. 
- 3 types of tokenizers in NLTK
  - word_tokenize()
  - wordpunct_tokenize()
  - sent_tokenize()
- when a tokenization is performed, we get individual tokens. sometimes it is necessary to group multiple tokens into 1.
  - Unigrams: "Steve" "went" "to" "school"
  - Bigrams: tokens of two consequtive words in a sentence; "Steve went" "went to" "to school"
  - Trigrams: tokens of 3; "Steve went to" "went to school"
  - Ngrams: tokens of n

Setting the stage for tokenization:
```python
import nltk
nltk.download('punkt')
text="In a world where technological advancements continue to redefine the boundaries of what is possible, the rapid integration of artificial intelligence, machine learning, and data-driven decision-making processes across industries ranging from healthcare, finance, and entertainment to education, agriculture, and manufacturing has opened up a plethora of opportunities for businesses, governments, and individuals to not only optimize their operations but also drive innovation in ways that were previously unimaginable, thus creating an ecosystem where collaboration between humans and machines can lead to transformative solutions that address complex global challenges such as climate change, poverty, and public health crises, while also ensuring that ethical considerations, regulatory frameworks, and the need for transparency remain at the forefront of this new era of technological evolution."
ml_tokens = nltk.word_tokenize(text)
list(nltk.bigrams(ml_tokens)) # or trigrams
```
### Parts of speech tagging & Stop words removal
- Parts of speech tagging: process of marking words as corresponding to parts of speech, based on both definition and context.
  - e.g. I like(Verb) to read(Verb) books
  - this is helpful in understanding the context in which a word is used.
- stopwords
  - e.g. a, the, is, are
  - not adding any important information, which can be elimiated.

```python
ml_tokens=nltk.word_tokenize("Jerry eats a banana")
nltk.download("averaged_perceptron_tagger") # needs to be downloaded for tagging.
for token in ml_tokens:
  print(nltk.pos_tag([token]))
```
This outputs
```python
[('Jerry', 'NN')]
[('eats', 'NNS')]
[('a', 'DT')]
[('banana', 'NN')]
```
pos_tag is a very basic version of the library, see how Jerry and eats is NNS - its tagged as a single term and categorized it as a Noun. In real life we use pos_tag from spacial library or transformers.

### Regular expression tokenizer
```python
from nltk.tokenize import RegexpTokenizer
sent = "Jerry eats a banana"
reg_tokenizer = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')
tokens = reg_tokenizer.tokenize(sent)
for token in tokens:
  print(nltk.pos_tag([token]))
```

```python
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
text="In a world where technological advancements continue to redefine the boundaries of what is possible, the rapid integration of artificial intelligence, machine learning, and data-driven decision-making processes across industries ranging from healthcare, finance, and entertainment to education, agriculture, and manufacturing has opened up a plethora of opportunities for businesses, governments, and individuals to not only optimize their operations but also drive innovation in ways that were previously unimaginable, thus creating an ecosystem where collaboration between humans and machines can lead to transformative solutions that address complex global challenges such as climate change, poverty, and public health crises, while also ensuring that ethical considerations, regulatory frameworks, and the need for transparency remain at the forefront of this new era of technological evolution."
ml_tokens=nltk.word_tokenize(text)
filtered_data = [w for w in ml_tokens if not w in stop_words]
filtered_data
```

### Stemming, Lemmatization
#### Stemming
Reducing a word or part of a word to its stem or root form. It lowers the inflection (process we do inorder to modify the word in order to communicate mini-gramatical categories like tensors, voices, aspect, gender, mood etc. added to communicate to other person) of words into their root form. This is a pre-processing activity.
Using the same word in different inflected forms in a text can lead to redundancy in natural language processing tasks. By reducing inflection, we decrease the number of unique words that machine learning models need to process.

**Example 1**
* Without Inflection: Original sentence: "She runs every day, and they are running in the park while he ran yesterday."
Inflected forms: runs, running, ran
* With Reduced Inflection:Simplified sentence: "She run every day, and they run in the park while he run yesterday." In this simplified version, we use "run" for all forms.
* Impact: Original sentence has three different inflected forms, which can create redundancy for a natural language processing model.
Simplified sentence reduces the variety of words, making it easier for the model to analyze the core action (running) without getting bogged down by different forms.

**Example 2**

* after stemming Generate → Generat also Generation → Generat
* Stemming can create non-dictionary forms (like "generat"). It's important to note that in stemming, the goal is to reduce words to their root form, which might not always be a valid dictionary word. The main purpose of stemming is to reduce data redundancy by grouping related words together. The primary aim is to reduce the variety of word forms to improve processing efficiency and analysis.

**Uses:**
- SEO
- Text mining
- Web-search
- Indexing
- Tagging.

**4 Types of Stemming Algorithms:**
* Porter Stemmer: Martin Porter invented it and Original Stemmer algorithm. Ease of use and rapid. 
* Snowball Stemmer: Also invented by same guy. more presise than porter stemmer.
* Lancaster Stemmer: Sometimes does over stemming, sometimes non linguistic or meaningless. 
* Regex Stemmer: morphological affixes.

```python
from nltk.stem import PorterStemmer
porter = PorterStemmer()
words = ['generous', 'generation', 'genorously','generate']
for word in words:
  print(f"{word} -> {porter.stem(word)}")
# Output
# generous -> gener
# generation -> gener
# genorously -> genor
# generate -> gener
from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language='english')
words = ['generous', 'generation', 'genorously','generate']
for word in words:
  print(f"{word} -> {snowball.stem(word)}")
# Output
# generous -> generous
# generation -> generat
# genorously -> genor
# generate -> generat
from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()
words = ['generous', 'generation', 'genorously','generate']
for word in words:
  print(f"{word} -> {lancaster.stem(word)}")
# Output
# generous -> gen
# generation -> gen
# genorously -> gen
# generate -> gen
from nltk.stem import RegexpStemmer
regex = RegexpStemmer('ing|s$|able$',min=4)
words = ['generous', 'generation', 'genorously','generate']
for word in words:
  print(f"{word} -> {regex.stem(word)}")
#Output
# generous -> generou
# generation -> generation
# genorously -> genorously
# generate -> generate
```
#### Lemmatization
Converting the words into root word using Parts of Speech (POS) tag as well as context as a base. Similar to stemming but brings context to the words and the result is a word in the dictionary. 
* Applications e.g. search engine and compacting
**Example:**
* eats → eat
* ate → eat
* ate → eat
* eating → eat

```python
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemma = WordNetLemmatizer()
words = ['generous', 'generation', 'genorously','generate']
for word in words:
  print(f"{word} -> {lemma.lemmatize(word)}")
# Output
# generous -> generous
# generation -> generation
# genorously -> genorously
# generate -> generate
```

# Named Entity Recognition
* First step in the information extraction
* NER seeks to locate and classify named entities into pre-defined categories such as names of person, Organization, location etc. e.g. Modi, America, Apple Inc, Tesla

## Challenges:
- Word sense disambiguiation: method by which meaning of the word is determined from the context it is used.
- Example: bark, cinnamon bark or sound made by dog is bark.
- when the two sentences passed to the algorithm word sense disambiguiation comes into picture, it removes the ambiguity. 

## Application:
* Text mining
* Information extraction
* used alongside with Lexicography
* Information retrieval process

## Algorithm:
**Lesk Algorithm:** based on the idea that words in each region will have a similar meaning.
```python
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
a1=lesk(word_tokenize('The building has a device to jam the signal'), 'jam')
print(a1, a1.definition())
a2=lesk(word_tokenize('I am stuck in a traffic jam'), 'jam')
print(a2, a2.definition())
a3=lesk(word_tokenize('I like to eat jam with bread'), 'jam')
print(a3, a3.definition())
#Output
# Synset('jamming.n.01') deliberate radiation or reflection of electromagnetic energy for the purpose of disrupting enemy use of electronic devices or systems
# Synset('jam.v.05') get stuck and immobilized
# Synset('jam.v.06') crowd or pack to capacity --> Somehow this isn't coming correct
```

## Named Entity Recognition
```python
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
text="Apple is an American company based out of California"
for w in nltk.word_tokenize(text):
  for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(w))):
    if hasattr(chunk, 'label'):
      print(chunk.label(), ' '.join(c[0] for c in chunk))
# Output - GPE stands for Geo political entity
# GPE Apple
# GPE American
# GPE California
``` 
# spaCy Library
spaCy is a free open source library for advaned Natural Language Processing in python for production use. NLTK was for research purpose. spaCy is for production use. Can handle and process large volume of text.
## Features
- Tokenization 
- Parts of Speech Tagging - word types of tokens, like verb or noun.
- Dependency Parsing
- Lemmatization
- Sentence Boundary Detection (SBD) - finding and segmenting individual sentences.
- Named Entity Recognition
- Entity Linking (EL)- Disambiguating texual entities to unique identifiers ina knowledge base.
- Similarity - comparing words, text apans and documents and how similar they are to each other.
- Text Classification - assigning caterfores or labels to a whole document or parts of it.
- Rule based Matching - finding sequence of token based on their texts and linguistic annotations, similar to regular expressions.
- Training - updating and improving a statstical models predictions
- Serialization - Saving objects to files or byte string.
```python
import spacy
nlp = spacy.load("en_core_web_sm") # verson of spacy library - english small model
doc = nlp("Apple is looking at buying U.K startup for $1 billion") # by default the spacy applies tagger, parser, ner

for token in doc:
  print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

# Output
# Apple Apple PROPN NNP nsubj Xxxxx True False
# is be AUX VBZ aux xx True True
# looking look VERB VBG ROOT xxxx True False
# at at ADP IN prep xx True True
# buying buy VERB VBG pcomp xxxx True False
# U.K U.K PROPN NNP dobj X.X False False
# startup startup VERB VB dep xxxx True False
# for for ADP IN prep xxx True True
# $ $ SYM $ quantmod $ False False
# 1 1 NUM CD compound d False False
# billion billion NUM CD pobj xxxx True False
```
* in the above by default the spacy applies tagger, parser, ner. The steps however can be added or replaced.
![](https://spacy.io/images/pipeline.svg)

* first step is tokenization
![](https://spacy.io/images/tokenization.svg)

```python
text="Mission impossible is one of the best movies I have watched. I love it."
print("{:10}|{:15}|{:15}|{:10}|{:10}|{:10}|{:10}|{:10}".format("text", "lemmatization", "partofspeech", "TAG", "DEP", "SHAPE", "ALPHA", "STOP"))
doc = nlp(text)
for token in doc:
  print("{:10}|{:15}|{:15}|{:10}|{:10}|{:10}|{:10}|{:10}".format(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop))

# text      |lemmatization  |partofspeech   |TAG       |DEP       |SHAPE     |ALPHA     |STOP      
# Mission   |mission        |NOUN           |NN        |nsubj     |Xxxxx     |         1|         0
# impossible|impossible     |ADJ            |JJ        |amod      |xxxx      |         1|         0
# is        |be             |AUX            |VBZ       |ROOT      |xx        |         1|         1
# one       |one            |NUM            |CD        |attr      |xxx       |         1|         1
# of        |of             |ADP            |IN        |prep      |xx        |         1|         1
# the       |the            |DET            |DT        |det       |xxx       |         1|         1
# best      |good           |ADJ            |JJS       |amod      |xxxx      |         1|         0
# movies    |movie          |NOUN           |NNS       |pobj      |xxxx      |         1|         0
# I         |I              |PRON           |PRP       |nsubj     |X         |         1|         1
# have      |have           |AUX            |VBP       |aux       |xxxx      |         1|         1
# watched   |watch          |VERB           |VBN       |relcl     |xxxx      |         1|         0
# .         |.              |PUNCT          |.         |punct     |.         |         0|         0
# I         |I              |PRON           |PRP       |nsubj     |X         |         1|         1
# love      |love           |VERB           |VBP       |ROOT      |xxxx      |         1|         0
# it        |it             |PRON           |PRP       |dobj      |xx        |         1|         1
# .         |.              |PUNCT          |.         |punct     |.         |         0|         0
# I you do not understand sonething
print(spacy.explain('nsubj')) #nominal subject
print(spacy.explain('pobj')) #object of preposition
# print entities
```
* Extracting the Named Entities
```python
text="Narendra Modi is the PM of India which is a country in the continent of Asia"
doc = nlp(text)
for token in doc.ents:
  print(token)
# Output
# Narendra Modi
# India
# Asia
```

* If you want to see a colourful version of the named entities then,
```python
from spacy import displacy
text="Narendra Modi is the PM of India which is a country in the continent of Asia which embraces Machine Learning"
doc=nlp(text)
displacy.render(docs=doc, style="ent",jupyter=True)
spacy.explain('GPE') #Geo Political Entity
```

# NLP Text Vectorization
Convertion of raw text into numerical form is called Text Vectorization. Machine learning expects text in numerical form. This is also called Feature Extraction.
Many ways of achieving feature extraction:
1. One Hot Encoding
2. Count Vectorizer
3. TF-IDF
4. Word Embeddings

## One Hot Encoding
Every word including symbols are written in the vector form. This vector will only have 0 & 1s. each word is written or encoded as a one hot vector, each word will have different vector representation. example:

| Color  | Red | Blue | Green |
|--------|-----|------|-------|
| Red    |  1  |  0   |   0   |
| Blue   |  0  |  1   |   0   |
| Green  |  0  |  0   |   1   |
| Red    |  1  |  0   |   0   |
| Green  |  0  |  0   |   1   |

```python
corpus = ['dog eats meat','man eats meat']
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
all_in_one = [indi.split() for indi in corpus]
one_hot.fit_transform(all_in_one).toarray()
#Output
# [['dog', 'eats', 'meat'], ['man', 'eats', 'meat']]
# array([[1., 0., 1., 1.],
#        [0., 1., 1., 1.]])
```

we generally dont use the scikitlearn onehotencoding directly as it's mainly for structured data not for unstructured data.

### Disadvantages
* Size of the one hot encoding is propotional to the size of the vocabulary.
* Sparse representation of data
* Insufficent in storing, computing and learning from data.
* No sequence of words is considered and is ignored.
* If words outside the vocabulary exists there is no way to deal with it.
* Word context is not considered in the representation.

## Bag of Words technique (BoW)
NLP pipeline has multiple steps as mentioned above. This step comes in the feature engineering step. Classical text represenation technique. Representation of the text under the consideration of bag of words. Text is characterised by a unique set of words. e.g. movie was bad; movie was excellent. This is characterised by the unique set of words not based on where it occurs in the sentence. so if the word bad it will be in one bag and excellent it will be in a different bag.

**Application:** Sentiment analysis (positive and negative sentiments). Harry potter was good, a movie was good - they are classified into the same bag.

### Write your own Bow Representation
```python
# if you are adventrous and dont want to use the Count Vectorizer.
import pandas as pd
import re
t1 = "dog eats meat everyday!"
t2 = "mAn eats meat once in a while."
t3 = "man, eaTs dog rarely!!!"
sentences = [re.sub(r"[^a-zA-Z0-9]", " ", t1.lower()).split(), 
             re.sub(r"[^a-zA-Z0-9]", " ", t2.lower()).split(), 
             re.sub(r"[^a-zA-Z0-9]", " ", t3.lower()).split()]

all_words = [word for words in sentences for word in words] # return variable - then first for.. then second for
unique_words = set(all_words)

def bow(all, sentences):
  results = []
  for sentence in sentences:
    result = {word: 0 for word in all}
    for word in sentence:
      result[word] = 1
    results.append(result)
  print(pd.DataFrame(results))

bow(all_words, sentences)
# Output
# dog  eats  meat  everyday  man  once  in  a  while  rarely
# 0    1     1     1         1    0     0   0  0      0       0
# 1    0     1     1         0    1     1   1  1      1       0
# 2    1     1     0         0    1     0   0  0      0       1
```
### Disadvantages:
* Size of the vector increases with the size of the vocabulary
* Sparsity (property of being scattered) is still an issue.
* Does not capture the similarity between words (not context aware). 'I eat', 'I ate', 'I ran' Bag of Words Vectors for all the three documents will be equally apart - in layman terms - 'eat and ran' and 'eat and ate' will be same distance apart.

```python
# use the countvectorize or just write your own python code after finding the unique words
from sklearn.feature_extraction.text import CountVectorizer
import re
t1 = "dog dog dog dog, dog eats meat everyday!"
t2 = "mAn eats meat once in a while."
t3 = "man, eaTs dog rarely!!!"
sentences = [re.sub(r"[^a-zA-Z0-9]", " ", t1.lower()), 
             re.sub(r"[^a-zA-Z0-9]", " ", t2.lower()), 
             re.sub(r"[^a-zA-Z0-9]", " ", t3.lower())]
all_words = [word for words in sentences for word in words] # return variable - then first for.. then second for
unique_words = set(all_words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
bag_of_words = X.toarray()
feature_names = vectorizer.get_feature_names_out()
print("Feature Names (Vocabulary):", feature_names)
print("Bag of Words Representation:")
print(bag_of_words)
#Output
# Feature Names (Vocabulary): ['dog' 'eats' 'everyday' 'in' 'man' 'meat' 'once' 'rarely' 'while']
# Bag of Words Representation:
# Feature Names (Vocabulary): ['dog' 'eats' 'everyday' 'in' 'man' 'meat' 'once' 'rarely' 'while']
# Bag of Words Representation:
# [[5 1 1 0 0 1 0 0 0]
#  [0 1 0 1 1 1 1 0 1]
#  [1 1 0 0 1 0 0 1 0]]
# See the count of 5 against dog. it not only counts it also describes it
```

Now even if you want it as a unigram, bigram and trigram thats also possible.
```python
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
t1 = "dog dog dog dog, dog eats meat everyday!"
t2 = "mAn eats meat once in a while."
t3 = "man, eaTs dog rarely!!!"
sentences = [re.sub(r"[^a-zA-Z0-9]", " ", t1.lower()), 
             re.sub(r"[^a-zA-Z0-9]", " ", t2.lower()), 
             re.sub(r"[^a-zA-Z0-9]", " ", t3.lower())]
all_words = [word for words in sentences for word in words] 
unique_words = set(all_words)
vectorizer = CountVectorizer(ngram_range=(1,3)) # See here <--
X = vectorizer.fit_transform(sentences)
bag_of_words = X.toarray()
feature_names = vectorizer.get_feature_names_out()
print("Feature Names (Vocabulary):", feature_names)
print("Bag of Words Representation:")
pd.DataFrame(bag_of_words)
#Output
# Feature Names (Vocabulary): ['dog' 'dog dog' 'dog dog dog' 'dog dog eats' 'dog eats' 'dog eats meat'
#  'dog rarely' 'eats' 'eats dog' 'eats dog rarely' 'eats meat'
#  'eats meat everyday' 'eats meat once' 'everyday' 'in' 'in while' 'man'
#  'man eats' 'man eats dog' 'man eats meat' 'meat' 'meat everyday'
#  'meat once' 'meat once in' 'once' 'once in' 'once in while' 'rarely'
#  'while']
# Bag of Words Representation:
# 0	1	2	3	4	5	6	7	8	9	...	19	20	21	22	23	24	25	26	27	28
# 0	5	4	3	1	1	1	0	1	0	0	...	0	1	1	0	0	0	0	0	0	0
# 1	0	0	0	0	0	0	0	1	0	0	...	1	1	0	1	1	1	1	1	0	1
# 2	1	0	0	0	0	0	1	1	1	1	...	0	0	0	0	0	0	0	0	1	0
# 3 rows × 29 columns
```
### Pros and Cons
- has the ability to capture the context and word order information in the form of n-grams
- Documents  having the same ngrams will have vectors closer to each other in euclidean space as compared to documents with different ngrams.
- As n increased the dimensionlity (sparsity) increases
- issue related to out of vocabulary problem exists

