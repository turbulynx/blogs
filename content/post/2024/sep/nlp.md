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
- Data collection
- Text Cleaning
- Pre-processing
- Feature engineering
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
