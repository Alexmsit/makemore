import torch 


# init generator for deterministic behaveiour
g = torch.Generator().manual_seed(1111) 

# load data and prepare conversion between string and integer
words = open("names.txt", "r").read().splitlines()
unique_chars = sorted(list(set("".join(words))))           
stoi = {s:i+1 for i,s in enumerate(unique_chars)}          
stoi["."] = 0 
itos = {i:s for s,i in stoi.items()}

# init the network
N = torch.zeros((27, 27), dtype=torch.int32)    # this matrix holds the occurences of each bigram pair dims = [27, 27]

# create the dataset
for word in words:                               
    chars = ["."] + list(word) + ["."]          # a special token is added at the beginning and the end of each sequence (which is a single word in this case)
    for char1, char2 in zip(chars, chars[1:]):  # ch1 is the first char of the bigram and ch2 is the second char
        ix1 = stoi[char1]                       # the strings are converted into integers
        ix2 = stoi[char2]
        N[ix1, ix2] += 1                        # the value of the counter of the bigram pair at the given indices is incremented by 1
P = (N+1).float()                               # the occurence matrix is converted to float
P = P / P.sum(dim=1, keepdim=True)              # the probability of each bigram pair is calculated by normalizing with the sum over the row (which are all possible bigram pairs starting with char1)

# generate predictions with the model
preds = []
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
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
