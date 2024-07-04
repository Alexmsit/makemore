import torch
import torch.nn.functional as F
import random

from utils.build_dataset import build_dataset



def main():

    # fix seeds for deterministic behaveiour
    g = torch.Generator().manual_seed(1111) 
    random.seed(1111)

    # load data, create mapping between strings and integers
    words = open("data/names.txt", "r").read().splitlines()         # store data in list, each element is one string
    unique_chars = sorted(list(set("".join(words))))                # get all unique characters of the data
    stoi = {s:i+1 for i,s in enumerate(unique_chars)}               # create string to int mapping       
    stoi["."] = 0                                                   # add "." string at index 0 (this char symbolizes the beginning and end of a sequence)
    itos = {i:s for s,i in stoi.items()}                            # create int to string mapping

    # set context window (how many chars does the model take to predict the next one?)
    block_size = 3      

    # create the dataset splits
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    X_train, Y_train = build_dataset(words[:n1], stoi, block_size)  # shape = [N,3,10]
    X_val, Y_val = build_dataset(words[n1:n2], stoi, block_size)    
    X_test, Y_test = build_dataset(words[n2:], stoi, block_size)

    # create model (Embedding and two linear layers (MLP))
    C = torch.randn((27,10), generator=g, requires_grad=True)
    W1 = torch.randn((30,200), generator=g, requires_grad=True)
    b1 = torch.randn(200, generator=g, requires_grad=True)
    W2 = torch.randn((200, 27), generator=g, requires_grad=True)
    b2 = torch.randn(27, generator=g, requires_grad=True)
    params = [C, W1, b1, W2, b2]

    # train loop
    loss_step = []
    steps = []
    print("Start training of the MLP model...")
    for i in range(10000):

        # create minibatch with randomly selected data
        ix = torch.randint(0, X_train.shape[0], (32,))

        # forward pass                   
        emb = C[X_train[ix]].view(-1, 30)       # embed the input data as vector (shape = [N,30])
        h = torch.tanh(emb @ W1 + b1)           # [N,30] @ [30,200] = [N,200]
        logits = h @ W2 + b2                    # [N,200] @ [200,27] = [N,27]
        loss_train = F.cross_entropy(logits, Y_train[ix])   # calculate softmax and negative log liklihood (with high efficiency)
        
        # backward pass
        for p in params:
            p.grad = None
        loss_train.backward()

        # update
        for p in params:
            p.data += -0.1 * p.grad

        # track stats
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