---
title: Machine Learning thought process.
date: 2024-08-10
tags: ["Engineering Team", "Machine Learning"]
image : "/img/posts/machine-leaning-thinking.jpeg"
Description: "Machine Learning Tools and Algorithms at your disposal to Address Customer Use Cases during a Sales Calls"
---
# Objective
This blog will introduce the basic concepts of machine learning, offering a high-level understanding of different categories and algorithms. It aims to give the Engineering Team a broad view of machine learning, enabling them to effectively participate in sales and requirement-gathering discussions with our customers. However, this is not intended for our data science team, as they will receive a more detailed exploration of these topics separately.

# AI v/s ML:
![Copied from https://www.researchgate.net/](/blogs/img/posts/machine-learning-types.png)
Source: https://www.researchgate.net/
# Types of Machine learning based on amount of supervision:
## Supervised Machine Learning
---
Uses labeled data to train models to make predictions or classify data. This type of data has input and known outcomes.
### Regression
Predicts continuous outcomes based on input features. E.g house prices in Singapore based on historical data.
### Classification
Categorizes data into predefined classes or labels.
## Unsupervised Machine Learning
---
Analyzes unlabeled data to find hidden patterns or structures. Here we have a lot of raw data that is hard to find a pattern yet.
### Clustering
Grouping data into clusters of similar items. e.g. MNIST the data can be clustered into groups based on number. In our case the tagging exercise was one such example of clustering by unsupervised learning.
### Anomaly detection
Identifies rare or unusual data points that deviate from the norm. Share prices that were out of the oridinary like in case of Gamestop stock or lump detection.
### Dimentionality or Feature Reduction
Reduces the number of input features while retaining the essential information. e.g. by means of removing unwanted features or by combining features into one where possible.
### Association Rule Learning
Identifies relationships or patterns between variables in large datasets, our example - recommending the next product a customer might buy along with a current purchase.
## Semisupervised Machine Learining
---
Combines a small amount of labeled data with a large amount of unlabeled data to improve learning accuracy. In our case tagging could have been solved using semisupervised learning. Tagging in google photos people is another example.
## Reinforcement Machine Learning
---
Trains models to make sequences of decisions by rewarding or penalizing actions to achieve a goal. Computer playing pacman using Machine learning is an example.

# Batch v/s Online Learning:
## Batch Training
### What is it?
Creating model in one go. This involves usually creating trained model on your local machine and using the model to run on a server. 
### When is it used?
- Learn model once Predict many times using that model.
- When the dataset is large.
- when the algorithm involves huge computing power e.g. Image classification using Convolutional Neural Networks.
### Example
- Dog and Cat image Classification
## Online Training
### What is it?
Training in incremental steps. Usually trained on the server (not on local)
### When is it used?
- Continously learn and predict. When you want the Prediction to go hand in hand with incremental Learning 
- This is usually used when the data set is evolving this is also called **Concept drift**.
- When you have to train on the server with very large dataset.
- Limitated resources or expensive hardware
### Examples
- netflix recommendation engine incrementally learn with new movies
- email spam identification incremental learning
### Algorthms that let you do partial or online learning
- [River - library for streaming data](https://github.com/online-ml/river)
- [scikit learn - partial_fit](https://scikit-learn.org/0.15/modules/scaling_strategies.html)
- [vowpal wabbit](https://vowpalwabbit.org/)
# Set of Algorithms 
---
# Types of learning - Instance based learning v/s Model based Learning
---