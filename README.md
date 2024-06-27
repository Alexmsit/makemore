# makemore

## Info

This repository contains the reproduction of the [spelled-out into to Large Language Modeling](https://www.youtube.com/watch?v=PaCmpygFfXo&t=5s) by Andrej Karpathy.
The tutorial shows step by step how makemore, an autoregressive character-level language model, is built. 
The model takes one text file as input and is able to generate text that has the same linguistic form.
The dataset which is used in this project contains the 30000 most popular names of 2018 according to the us government.
Therefore, the output of the model are strings which sound like real names, but are not existing.
Under the hood, several different models, which are explained below, can be used to generate the output.

## Basic Bigram Language Model

The basic character-level bigram language model works with statistical values to generate predictions. In this case, a bigram contains two following characters of a given string, which means each name within the dataset contains multiple bigrams. For example, the name "Adam" naturally contains the following bigrams: ["Ad", "da", "am"]. Additionally, a special token (".") is inserted at the beginning and the end of the string, which leads to the following bigrams for the name "Adam": [".A", "Ad", "da", "am", "m."].

In the first step, all unique characters of the names within the dataset are saved. Then an empty 2D-matrix is created, the dimensions are defined by the number of unique characters within the dataset. Additionally a special token is added to symbolize the beginning and the end of a sequence (dims: num_unique_chars+1, num_unique_chars+1).

In the second step, the number of occurences of all bigram pairs of each name within the dataset are counted and stored within the matrix. Then each count is incremented by 1 for model smoothing (this is necessary because if a bigram within the dataset is counted 0 times, the liklihood will be inf)

In the third step, each of the counts get normalized to get the probability of the bigram pair. For normlaization, each count gets divided by the total sum of all possible bigrams starting with the current character (row of the matrix).

In the last step, the model can generate random predictions from this probability matrix. The generated names depend on the probabilities of the bigrams within the dataset and can be evaluated with the negative logarithmic liklihood. For this metric, the log of each probability is calculated and summed up. Lastly this value gets inversed, which means the possible range of this metric stretches from 0 to inf, while 0 indicates a perfect model and a higher values indicates a worse model.

## Neural Network Bigram Language Model




