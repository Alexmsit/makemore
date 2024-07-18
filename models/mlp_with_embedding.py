import sys
sys.path.append('../')
import random

import torch
import torch.nn.functional as F

from utils.build_dataset import load_data, build_dataset



def main(): 

    # fix seeds for deterministic behaveiour
    g = torch.Generator().manual_seed(1111) 
    random.seed(1111)
    
    # load data, vocab_size and mapping between int and str
    words, vocab_size, stoi, itos = load_data("../data/names.txt")

    # set parameters of model
    block_size = 3  # context window (how many chars does the model take to predict the next one?) 
    n_embd = 10     # dimensionality of the character embedding vectors
    n_hidden = 200  # number of neurons in the hidden layer

    # create model (Input-Embedding + 2-Layer-MLP)
    C = torch.randn((vocab_size, n_embd), generator=g)  
    W1 = torch.randn((n_embd*block_size, n_hidden), generator=g)    # the weights of the first layer get squashed so that the output of the tanh is not saturated anymore
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn((n_hidden, vocab_size), generator=g)         # the weights of the last layer of the MLP get squashed to minimize the loss
    b2 = torch.randn(vocab_size, generator=g)                     # initial loss went from ~27 to ~3 with this init because the logits get smaller, there is no "hockey-stick" loss curve anymore
    params = [C, W1, b1, W2, b2]

    for param in params: 
        param.requires_grad = True

    # create the dataset splits
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))
    
    X_train, Y_train = build_dataset(words[:n1], stoi, block_size)  # shape = [N,3,10]
    X_val, Y_val = build_dataset(words[n1:n2], stoi, block_size)    
    X_test, Y_test = build_dataset(words[n2:], stoi, block_size)

    # train loop
    batch_size = 32
    max_steps = 200000
    steps = []
    loss_step = []
    print("Start training of the MLP model...")
    for i in range(max_steps):

        # create minibatch with randomly selected data
        ix = torch.randint(0, X_train.shape[0], (batch_size,))
        X_train_batch = X_train[ix]
        Y_train_batch = Y_train[ix]

        # forward pass
        emb = C[X_train_batch]                  # embed the input data as vector (shape=[N,3,10])                   
        emb = emb.view(emb.shape[0], - 1)       # reshape the data (shape=[N,30])
        h = torch.tanh(emb @ W1 + b1)           # [N,30] @ [30,200] = [N,200]
        logits = h @ W2 + b2                    # [N,200] @ [200,27] = [N,27]
        loss_train = F.cross_entropy(logits, Y_train_batch)   # calculate softmax and negative log liklihood (with high efficiency)
        
        # backward pass
        for p in params:
            p.grad = None
        loss_train.backward()

        # update
        for p in params:
            p.data += -0.1 * p.grad

        # track stats
        if i % 1000 == 0:
            print(f"{i:7d}/{max_steps:7d}: {loss_train.item():.4f}")
        loss_step.append(loss_train.item())
        steps.append(i)
    
    print(f"Training finished, loss is at {loss_train:.4f}\n")

    # validation loop
    print("Start validation of the MLP model...")
    for i in range(1):
                         
        emb = C[X_val].view(-1, 30)       
        h = torch.tanh(emb @ W1 + b1)           
        logits = h @ W2 + b2                    
        loss_val = F.cross_entropy(logits, Y_val)
    
    print(f"Validation finished, loss is at {loss_val}.\n")

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
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)           
            logits = h @ W2 + b2                    
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