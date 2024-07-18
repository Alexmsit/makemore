# Makemore

<hr>

## 1. Info

This repository contains code from the [spelled-out into to Large Language Modeling](https://www.youtube.com/watch?v=PaCmpygFfXo&t=5s) by Andrej Karpathy as well as my own improvements on this code. The lecture shows how makemore, which is an autoregressive character-level language model, is built from scratch.

<hr>

## 2. Makemore introduction

In general, the makemore language model recieves text as input and is able to output text, which has the same linguistic form as the input text. In case of this lecture, the dataset consists of the 30000 most popular names of the year 2018 according to the us government. Therefore, the outputs of the model are strings, which can sound like real names, but might not exist.

<hr>

## 3. Implementation

The makemore model can be implemented in different ways. In the sections below, each of these methods is described in detail.

### 3.1 Basic Bigram Language Model

The most simple makemore model can be implemented with a basic character-level bigram language model, which works with statistical values from the training data to generate preditions. A character-level bigram is a sequence of two adjacent elements from a string. In the case of this lecture, this bigrams are built from the names within the dataset, where each name contains multiple bigrams. For example, the name "Adam" contains the following 3 bigrams: ["Ad", "da", "am"]. Additionally, a special token "." is inserted at the beginning and the end of each string, which leads to two additional bigrams for each name. In the case of "Adam" the resulting bigrams would be: [".A", "Ad", "da", "am", "m."].

Since the model only uses a statistical approach to generate predictions, there is no need for a training. However, these statisics need to be calculated first. In this case, we count all bigram pairs which occur within the training data and store these counts in a matrix. Since we have 26 unique characters as well as a special token, the dimensions of this matrix are [27,27], where each element holds the count of one bigram pair. Next, the row-wise sum of all counts is calculated and each count of the row is divided by this sum. This results in a probability distribution for the next character given a current character. All probabilities of a single row sum up to 1.

Lastly, the model can generate random predictions from this probability matrix. The generated strings are then evaluated with the negative log likelihood. For this metric, the log of each probability score of all single generated characters is calculated, summed up and divided by the number of characters (mean) to get the average log liklihood. Lastly, this value gets inversed so that the possible range reaches from 0 to inf, where 0 indicates a perfect model and a higher value indicates a worse model.

### 3.2 Basic Trigram Language Model

An extension of the basic character-level bigram language model is the trigram language model. The only difference to the bigram model of the last section is, that this model works with character-level trigrams. A character-level trigram is a sequence of three adjacent elements from a string. For example, the name "Adam" contains the following 4 trigams: [".Ad", "Ada", "dam", "am."].

Apart from using trigrams, the method uses the same statistical approach to generate predictions. In this case the matrix, which holds the counts of the occurences of each trigram has the shape [27,27,27], where each element holds the count of a trigram. The evaluation also uses the calculation of the average log likelihood as described in the last section.

### 3.3 Single-Layer Neural Network Bigram Language Model

A more complex way of implementing makemore is the usage of a character-level single-layer neural network bigram language model. This model works with a single-layer neural network, which learns representations from the training data to generate predictions. Instead of creating a matrix with statistical probabilities from the training data as seen in the last two sections, this model learns probabilities through optimization of the paramters of the model. While this weight matrix has the same shape as the matrix in the first section, the paramters are randomly initialized and optimized through gradient descent.

Since this implementation of makemore contains a single-layer neural network, a training is necessary. Here the data is multiplicated with the weight matrix. This results in the logits of the model, which then get exponentiated and normalized (which is the Softmax function) to get the probabilities for each bigram pair. The evaluation also uses the calculation of the average log likelihood as described in the last section.

### 3.4 MLP Language Model with input embedding

Another similiar way of implementing makemore is the usage of an input embedding followed by a MLP. This model takes the input and embedds it as a high-dimensional vector. This representation is then fed into two linear layers to generate predictions. As explained in the last section, this model also learns representations from the training data by optimizing the parameters of the model through gradient descent. Here the paramteres are from the two weight matrices, their biases and the input embedding.

Since we have an input embedding and two linear layers, a training is necessary. Here the input is embedded in a high-dimensional vector, then this data is multiplicated with the weight matrices of the two linear layers to generate the logits. The parameters of the linear layers as well as the input embedding are optimized. This logits then get exponentiated and normalized (Softmax) to get the probabilities of the next character. The evaluation also uses the calculation of the average log likelihood as described above. 

<hr>

## 4. Results

|  Model | Loss  |
|---|---|
| Bigram   | 2.5139  |
| Trigram  | 2.3312  |
| Single-Layer NN |   |
| MLP  |   |