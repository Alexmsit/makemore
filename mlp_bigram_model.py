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

# create the dataset
block_size = 3                  # this is the context length: how many chars to predict the next one?
X, Y, = [],[]
for word in words:
    context = [0] * block_size                               
    for char in word + ".":             # iterate over all chars of the current word (special token is added at the end)
        ix = stoi[char]                 # the strings are converted into integers
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]    # shift elements of context to the right and add new last character

X = torch.tensor(X)
Y = torch.tensor(Y)

# create model (LUT for embedding and two linear layers (MLP))
C = torch.randn((27,2), generator=g, requires_grad=True)
W1 = torch.randn((6,100), generator=g, requires_grad=True)
b1 = torch.randn(100, generator=g, requires_grad=True)
W2 = torch.randn((100, 27), generator=g, requires_grad=True)
b2 = torch.randn(27, generator=g, requires_grad=True)
params = [C, W1, b1, W2, b2]

# create input embedding -> each character is embedded as a 2-dimensional vector
emb = C[X]                      # shape = [N,3,2]
emb = emb.view(-1, 6)           # shape = [N,6]

# forward pass
h = torch.tanh(emb @ W1 + b1)   # shape = [N, 100]
logits = h @ W2 + b2            # shape = [N, 27]
#counts = logits.exp()                                  
#prob = counts / counts.sum(1, keepdims=True)       # this can be done more efficient with torch.nn.functional.cross_entropy
#loss = -prob[torch.arange(32), Y].log().mean()
loss = F.cross_entropy(logits, Y)

# backward pass
for p in params:
    p.grad = None
loss.backward()

# update
for p in params:
    p.data += -0.1 * p.grad
