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
| Aspect                           | Model-Based Learning                                            | Instance-Based Learning                                       |
|----------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------|
| Data Preparation             | Prepare or preprocess the data for model training               | Prepare or preprocess the data for model training              |
| Training and Pattern Discovery | Train and create a model to discover patterns and then use it to predict | No training; pattern discovery is postponed until scoring query is received |
| Model Storage                | Model stored in a suitable form                                 | No model                                                       |
| Generalization               | Generalize the rules in the model before the scoring instance is seen | Generalizes for each scoring instance individually as and when seen |
| Prediction                   | Predict for unseen scoring instance using the model             | Predicts for unseen scoring instance using training data directly |
| Data Usage                   | Training data discarded after the model is trained              | Data used for each query - full set of training data is needed |
| Model Form                   | Requires known model form                                       | Model may not have an explicit form                             |
| Storage Space                | Generally storing the model requires less space                 | Storing training data requires more space                      |

# Customer Satisfaction:
Our customers are our top priority, and their satisfaction is our greatest reward. In this document, we outline common obstacles to delivering an exceptional customer experience, so we can proactively address and avoid them.

## Understanding the Common Reasons for Unsatisfied customer:
- Inaccurate results, check for:
    - Insufficient Data
    - Poorly Labelled Data
    - Non Representative Data
        - Sampling Noise
        - Sampling Bias
    - Poor Quality Data
    - Irrelevant Features
    - Overfitting
    - Underfitting
    - Inappropriate Evaluation Metrics
- Usability Issues
    - High Cost to Value ratio
    - Inability to integrate with existing systems
    - Slow response times
- Data Privacy concerns
    - Regulatory Requirements.
    - Ethical & Bias Concerns
- User Experience
    - Unclear customer objective
    - lack of transparency
    - over promising/under delivery
    - any knowledge gaps between engineering and data science teams.
- Communication 
    - communicate to customer in writing on what he is expecting before putting any effort.

# MLDLC (Machine Learning Development Life Cycle)
1. Frame the problem
2. Gathering Data, Data Clensing and Processing
    - Remove duplicates
    - Remove missing
    - Remove outliers
    - Scale the values to same base (like property value needs to consider inflation)
4. Exploratory Data Analysis (Input and output relationship study)
    - Vitualization
    - Univariate, Bi-Variate, Multivariate Analysis (outlier detection)
    - Imbalanced Data set (more dog photos less cat - how to handle?)
5. Feature Engineering and selection
6. Model Training Evaluation and selection
    - Ensemble learning - combines or aggregates multiple learnings into one
7. Model Deployment
8. Testing
9. Optimizing model

# Stages
1. Technology Map for EigenAI.
    - Algorithms & Datastructures
    - Java 
    - Python 
        - tensorflow
    - Node.
    - DBMS/SQL
    - NoSQL
    - Big Data Tools (Spark, Hadoop, Kafka, Hive)
    - Cloud Platforms (AWS, GCP)
    - Data Pipelines (Apache Airflow)
    - Statistical Programming
    - SQL
    - Advanced Excel
    - Distributed Systems
    - System Engineering & Systems Design

# Numpy
- Scalar (0D Tensor)
    - e.g. (2)
    ```python
    import numpy as np
    a = np.array(2)
    a # --> array(2)
    ```
- Vector (1D Tensor)
    - e.g. ([1,2,3,4])
    ```python
    a = np.array([1,2,3,4])
    ```
    this is a **1D vector with 4 dimentions.**. This is a collection of scalars. usecase - timeseries
- Matrices (2D Tensor)
    - this is a collection of vectors.
   ```python
    import numpy as np
    a=np.array([[1,2,3,4],[5,6,7,8]])
    a.shape # (2, 4)
    ```
    usecase - computer vision (videos)
- N Dimentional tensor.

[Lee Vaughan - Python Tools for Scientists_ An Introduction to Using Anaconda, JupyterLab, and Python's Scientific Libraries](https://drive.google.com/drive/folders/1gSIJH06PizsWb41jN4C0geukYSWN-R3n)

# Pandas
[Lee Vaughan - Python Tools for Scientists_ An Introduction to Using Anaconda, JupyterLab, and Python's Scientific Libraries](https://drive.google.com/drive/folders/1gSIJH06PizsWb41jN4C0geukYSWN-R3n)

# Where to find datasets
- [Kaggle](kaggle.com)
- 


CB*$6o]YdU|;/HRF