import torch
import torch.nn.functional as F


# init generator for deterministic behaveiour
g = torch.Generator().manual_seed(1111) 

# load data and prepare conversion between string and integer
words = open("data/names.txt", "r").read().splitlines()
unique_chars = sorted(list(set("".join(words))))           
stoi = {s:i+1 for i,s in enumerate(unique_chars)}          
stoi["."] = 0 
itos = {i:s for s,i in stoi.items()}      

# init the network
W = torch.randn((27,27), generator=g, requires_grad=True)   # this matrix holds the weights of the neural network

# create the dataset
data, labels = [], []
for word in words:
    chars = ["."] + list(word) + ["."]                      # a special token is added at the beginning and the end of each sequence (which is a single word in this case)
    for char1, char2 in zip(chars, chars[1:]):              # char1 is the first char of the bigram and char2 is the second char
        ix1 = stoi[char1]                                   # the strings are converted into integers
        ix2 = stoi[char2]
        data.append(ix1)                                    # the data and labels are appended to a list
        labels.append(ix2)
data = torch.tensor(data)                                   # data and labels are then converted to a torch.tensor
labels = torch.tensor(labels)
num = data.nelement()                                       # get the number of training examples

# train loop
for k in range(100):
    # forward pass
    data_enc = F.one_hot(data, num_classes=27).float()      # encode the data as 1-hot-encoding (which now has the dims [N, 27])
    logits = data_enc @ W                                   # calculate the product of weight matrix and data matrix (the whole dataset is processed in 1 step, no batches here)
                                                            # dims: [N,27] @ [27,27] = [N,27]
    counts = logits.exp()                                   # exponentiate the logits and normalize by the sum of counts over the whole row
    probs = counts / counts.sum(1, keepdims=True)           # normalize all counts by the sum over all counts of the corresponding row (this is the Softmax function, https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) 
    loss = -probs[torch.arange(num), labels].log().mean()   # calculate logarithmic probabilities, then calculate the mean of these probabilities and inverse it (results in average log liklihood)
    #print(f"Epoch {k} Loss: {loss.item()}")                 

    # backward pass
    W.grad = None                                           # reset gradients
    loss.backward()                                         # calculate backward pass
 
    # update
    W.data += -50 * W.grad                                   # update weights

# generate predictions with the model
preds = []
print("\nGenerating predictions with the neural net bigram model...\n")
for i in range(5):
    out = []
    ix = 0
    while True:
        data_enc = F.one_hot(torch.tensor([ix]), num_classes=27).float()  # encode the data as 1-hot-encoding (which now has the dims [1, 27]) 
        logits = data_enc @ W                               # calculate the product of weight matrix and the input data
        counts = logits.exp()                               # exponentiate the logits and normalize by the sum of counts over the whole row
        probs = counts / counts.sum(1, keepdims=True)       # normalize all counts by the sum over all counts of the corresponding row (this is the Softmax function, https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)

        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()  # sample index according to the probabilities
        out.append(itos[ix])                                # convert index to character and store character
        if ix == 0:                                         # if the characters is the special token, end the sequence
            break
    pred = "".join(out)                                     # join all chars to get the final predition           
    preds.append(pred)

# evaluate the predictions
log_likelihood = 0.0
n = 0
for i, pred in enumerate(preds):
    for char1, char2 in zip(pred, pred[1:]):
        ix1 = stoi[char1]                                   # the strings are converted into integers
        ix2 = stoi[char2]

        data_enc = F.one_hot(torch.tensor([ix1]), num_classes=27).float()  # encode the data as 1-hot-encoding (which now has the dims [1, 27]) 
        logits = data_enc @ W                               # calculate the product of weight matrix and input data
        counts = logits.exp()                               # exponentiate the logits and normalize by the sum of counts over the whole row
        probs = counts / counts.sum(1, keepdims=True)       # normalize all counts by the sum over all counts of the corresponding row (this is the Softmax function, https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) 
        prob = probs[0, ix2]                                # select the probability of the predicted next character
        log_prob = torch.log(prob)                          # calculate the logarithmic probability of the generated bigram pair

        log_likelihood += log_prob              # sum up the logarithmic probabilities of all bigrams of the generated prediction to get the log liklihood (equal to log of product of the single probabilities) 
        neg_log_likelihood = -log_likelihood    # invert the logarithmic likelihood (will be a positive value with range [0, inf])

        n += 1
    avg_log_likelihood = neg_log_likelihood / n # calculate the average logarithmic likelihood over all bigrams of the prediction (can be seen as a loss value)
    print(f"prediction: '{pred}'")
    print(f"Average log liklihood: {avg_log_likelihood:.4f}\n")

print(f"Average log liklihood over whole dataset: {loss:.4f}")
