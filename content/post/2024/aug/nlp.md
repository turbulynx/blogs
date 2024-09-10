---
title: Generative AI - NLP 
date: 2024-07-11
tags: ["Chatbots", "RASA", "ChatGPT", "BERT", "Transformers", "Prompt Engineering"]
image : "/img/posts/google-ads-integration.jpg"
Description  : "Generative AI with NLP LLM: 
"
---
# Objective

# Introduction
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
    - ![](syntax-parse-tree.png)
    - NP - noun phrase
    - VP - verb phrase
    - PP -
    - S - sentence at the highest level.
- **Context**: words and sentences that surround any part of discourse and that helps determine the meaning. Application: Sarcasm detection, summarization, topic modelling.
  - semantics: direct meaning 
  - pragmatics: adds world knowledge and external knowledge.

## Challenges of NLP
- Ambiguity: two or more meanings of a single passage. e.g. we saw her duck. Common knowledge assumptions. e.g he says Sun rises in the west (assumption that a preson knows sun rises in the east)
- Creativity

---
## NLTK library
NLTK library is most commonly used NLP library. Common text pre-processing steps in NLP: 
  - Text preprocessing for Tokenization
  - Stemming
  - Lemmatization
  - Word Embedding
  - Parts of speech tagging
  - Stop Word removal
  - Word Sence disambiguation
  - Named Entity Recognition (NER)
    
