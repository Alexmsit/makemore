import torch
import torch.nn.functional as F

# init generator for deterministic behaveiour
g = torch.Generator().manual_seed(1111) 

# load data and prepare conversion between string and integer
words = open("names.txt", "r").read().splitlines()
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
    for char1, char2 in zip(chars, chars[1:]):              # ch1 is the first char of the bigram and ch2 is the second char
        ix1 = stoi[char1]                                   # the strings are converted into integers
        ix2 = stoi[char2]
        data.append(ix1)                                    # the data and labels are appended to a list
        labels.append(ix2)
data = torch.tensor(data)                                   # data and labels are then converted to a torch.tensor
labels = torch.tensor(labels)
num = data.nelement()                                       # get the number of training examples

# train loop
print("start training over 100 epochs...\n")
for k in range(100):
    # forward pass
    data_enc = F.one_hot(data, num_classes=27).float()     # encode the data as 1-hot-encoding (this results in )
    logits = data_enc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), labels].log().mean()
    print(f"Epoch {k} Loss: {loss.item()}")

    # backward pass
    W.grad = None
    loss.backward()

    # update
    W.data += -0.1 * W.grad

# generate predictions with the model
preds = []
for i in range(5):
    out = []
    ix = 0
    while True:
        data_enc = F.one_hot(data, num_classes=27).float()     # encode the data as 1-hot-encoding (this results in )
        logits = data_enc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    pred = "".join(out)
    preds.append(pred)

# evaluate the predictions
log_likelihood = 0.0
n = 0
for i, pred in enumerate(preds):
    print(f"\nmodel prediction {i}: {pred}")
    for char1, char2 in zip(pred, pred[1:]):
        ix1 = stoi[char1]
        ix2 = stoi[char2]

        prob = P[ix1, ix2]
        log_prob = torch.log(prob)

        log_likelihood += log_prob
        neg_log_likelihood = -log_likelihood

        n += 1
    avg_log_likelihood = neg_log_likelihood / n
    print(f"Average log liklihood: {avg_log_likelihood:.4f}")
