# Makemore

<hr>

## 1. Info

This repository contains code as well as notes from the [spelled-out into to Large Language Modeling](https://www.youtube.com/watch?v=PaCmpygFfXo&t=5s) by Andrej Karpathy. This lecture shows how makemore, which is an autoregressive character-level language model, is built from scratch.

<hr>

## 2. Makemore introduction

In general, the makemore language model recieves text as input and is able to also output text, which has the same linguistic form as the input text. In case of this lecture, the dataset consists of the 30000 most popular names of the year 2018 according to the us government. Therefore, the outputs of the model are strings, which can sound like real names, but are not existing.

<hr>

## 3. Implementation

The makemore model can be implemented in different ways. In the sections below, each of these methods is described in detail.

### 3.1 Basic Bigram Language Model

The most simple makemore model uses a basic character-level bigram language model. This model works with statistical values from the training data to generate predictions. A character-level bigram contains two following characters of a string. In this case, the bigrams are built from the names of the dataset, where each name contains multiple bigrams. For example, the name "Adam" naturally contains the following bigrams: ["Ad", "da", "am"]. Additionally, a special token (".") is inserted at the beginning and the end of each string, which leads to the following bigrams for the name "Adam": [".A", "Ad", "da", "am", "m."].

In the first step, the unique characters of all names within the dataset are saved. Then an empty 2D-matrix is created, the dimensions are defined by the number of unique characters within the dataset. Additionally a special token is added to symbolize the beginning and the end of a sequence so the dimensions of the matrix are [num_unique_chars+1, num_unique_chars+1].

In the second step, the occurence of all bigram pairs within the dataset are counted and stored within the 2D-matrix. Then each count is incremented by 1 for model smoothing (this is necessary because if a bigram within the dataset is counted 0 times, the liklihood which will be calculated later will be inf).

In the third step, each of the counts get normalized to get the probability of the bigram pair. For normlaization, each count gets divided by the total sum of all possible bigrams starting with the current character (row of the 2D-matrix).

In the last step, the model can generate random predictions from this probability matrix. The generated names depend on the probabilities of the bigrams within the dataset and can be evaluated with the negative logarithmic liklihood. For this metric, the log of each probability is calculated and summed up. Lastly this value gets inversed, which means the possible range of this metric stretches from 0 to inf, where 0 indicates a perfect model and a higher value indicates a worse model.

### 3.2 Single-Layer Neural Network Bigram Language Model

Another way of implementing makemore is the usage of a character-level neural network bigram language model. This model works with a single-layer neural network, which learns representations from the training data to generate predictions. The input of the model as well as the output of the model is the same as explained in the last section.

In the first step, the unique characters of all names within the dataset are saved. Then a random weights 2D-matrix is created, the dimensions are defined by the number of unique characters within the dataset. Additionally a special token is added to symbolize the beginning and the end of a sequence so the dimensions of the matrix are [num_unique_chars+1, num_unique_chars+1].

In the second step, the model is trained on the data for a fixed number of epochs (100 in this case). During training, the data gets one-hot encoded and multiplicated with the weight matrix. This results in the logits of the model, which then get exponentiated and normalized (which is the Softmax function) to get the probabilities for each bigram pair.
The loss is also the negative logarithmic liklihood, which is calculated by building the log of the probabilities and summing up. Lastly this value gets inversed and the mean is calculated to get the negative logarithmic liklihood. The whole forward pass is differentiable, which means backpropagation can be used to optimize the weights with gradient descent, which is done in the backward pass.

In the last step the model can generate random predictions from the weight matrix, which holds the learned representation of the probabilities of the bigram pairs.



