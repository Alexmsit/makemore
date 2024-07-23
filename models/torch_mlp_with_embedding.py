import argparse
import sys
sys.path.append('../')
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from utils.build_dataset import load_data, build_dataset
from utils.get_statistics import get_activation_statistics



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Activate the debug mode and show statistics of the model.")
    parser.add_argument("--save_figs", action="store_true", help="Save the generated figures when in debug mode.")
    args = parser.parse_args()

    return args



class MLP(nn.Module):
    def __init__(self, vocab_size, emb_dim, block_size, n_hidden, args):
        super().__init__()

        self.emb_dim = emb_dim
        self.block_size = block_size
        self.debug = args.debug
        self.save_figs = args.save_figs

        if self.debug:
            self.activation_dict = {
                "Layer_1": [],
                "Layer_2": []
            }

        # embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

        # layer 1
        self.linear_1 = nn.Linear(in_features=block_size*emb_dim, out_features=n_hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden)
        self.tanh_1 = nn.Tanh()

        # layer 2
        self.linear_2 = nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=False)
        self.bn_2 = nn.BatchNorm1d(num_features=n_hidden)
        self.tanh_2 = nn.Tanh()

        # out layer
        self.linear_3 = nn.Linear(in_features=n_hidden, out_features=vocab_size, bias=False)
        self.bn_3 = nn.BatchNorm1d(num_features=vocab_size)    



    def forward(self, x, num_steps):

        x = self.embedding(x).view(-1, self.block_size*self.emb_dim)

        x = self.linear_1(x)
        x = self.bn1(x)
        x = self.tanh_1(x)

        if self.debug and (num_steps % 5000 == 0):
           self.activation_dict["Layer_1"].append((x, num_steps))

        x = self.linear_2(x)
        x = self.bn_2(x)
        x = self.tanh_2(x)

        if self.debug and (num_steps % 5000 == 0):
           self.activation_dict["Layer_2"].append((x, num_steps))
        
        x = self.linear_3(x)
        logits = self.bn_3(x)

        return logits



def main(): 

    args = parse_args()
    

    # fix seeds for deterministic behaveiour
    g = torch.Generator().manual_seed(1111) 
    random.seed(1111)
    
    # load data, vocab_size and mapping between int and str
    words, vocab_size, stoi, itos = load_data("../data/names.txt")

    # set parameters of model
    block_size = 3      # context window (how many chars does the model take to predict the next one?) 
    emb_dim = 10        # dimensionality of the character embedding vectors
    n_hidden = 200      # number of neurons in the hidden layer

    # create model
    model = MLP(vocab_size, emb_dim, block_size, n_hidden, args)
    model.train()
    
    # create optimizer
    optimizer = SGD(model.parameters(), lr=0.03)

    # create the dataset splits
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))
    
    X_train, Y_train = build_dataset(words[:n1], stoi, block_size)  # shape = [N,3,10]
    X_val, Y_val = build_dataset(words[n1:n2], stoi, block_size)    
    X_test, Y_test = build_dataset(words[n2:], stoi, block_size)

    # train loop
    batch_size = 32
    max_steps = 10000
    print("Start training of the MLP model...")
    for i in range(max_steps):

        # create minibatch with randomly selected data
        ix = torch.randint(0, X_train.shape[0], (batch_size,))
        X_train_batch = X_train[ix]
        Y_train_batch = Y_train[ix]

        # forward pass
        logits = model(X_train_batch, i)
        
        # loss
        loss_train = F.cross_entropy(logits, Y_train_batch)   # calculate softmax and negative log liklihood (with high efficiency)
        
        # backward pass
        optimizer.zero_grad()
        loss_train.backward()

        # update
        optimizer.step()

        # track stats
        if i % 1000 == 0:
            print(f"step: {i} | loss: {loss_train:.4f}")
    
    print(f"Training finished, loss is at {loss_train:.4f}\n")

    if args.debug:
        get_activation_statistics(model.activation_dict, args.save_figs, metrics=["single_hist", "multi_hist", "heatmap"])

    # validation loop
    print("Start validation of the MLP model...")
    model.eval()
    model.debug = False
    for i in range(1):
                         
        logits = model(X_val, 0)    

        loss_val = F.cross_entropy(logits, Y_val)
    
    print(f"Validation finished, loss is at {loss_val:.4f}.\n")

    # generate predictions with the model
    preds = []
    print("Generating predictions with the mlp model...\n")
    for j in range(5):

        n = 0               # counter for number of chars of each prediction
        log_likelihood = 0  # loss metric
        out = []
        context = [0] * block_size

        while True:

            # forward pass
            logits = model(torch.tensor(context), 0)  

            probs = F.softmax(logits, dim=1)        
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            
            # calculate loss
            prob = probs[0, ix]
            log_prob = torch.log(prob)
            log_likelihood += log_prob
            neg_log_likelihood = -log_likelihood

            # update context window
            context = context[1:] + [ix]

            # append char to list
            out.append(itos[ix])

            # increment counter
            n += 1

            # if special char is reached, break the loop
            if ix == 0:
                break
        
        # join all chars to get the final predition 
        pred = "".join(out)                                   
        preds.append(pred)

        # calculate average log likelihood for all chars of the prediction
        avg_log_likelihood = neg_log_likelihood / n
        print(f"prediction: '{pred}'")
        print(f"Average log liklihood: {avg_log_likelihood:.4f}\n")

    print(f"Average log liklihood over whole dataset: {loss_train:.4f}")



if __name__ == "__main__":
    main()