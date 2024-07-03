import torch 


# init generator for deterministic behaveiour
g = torch.Generator().manual_seed(1111) 

# load data and prepare conversion between string and integer
words = open("data/names.txt", "r").read().splitlines()
unique_chars = sorted(list(set("".join(words))))           
stoi = {s:i+1 for i,s in enumerate(unique_chars)}          
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

# init the network
N = torch.zeros((27, 27, 27), dtype=torch.int32)

# create the dataset/generate the counts and probabilities of each trogram pair
for word in words:                               
    chars = ["."] + list(word) + ["."]              # two special tokens are added at the beginning and the end of each sequence (which is a single word in this case)
    for char1, char2, char3 in zip(chars, chars[1:], chars[2:]):    # char1 is the first character of the trigram, char2 is the second character, char3 is the third character
        ix1 = stoi[char1]                       # the strings are converted into integers
        ix2 = stoi[char2]
        ix3 = stoi[char3]
        N[ix1, ix2, ix3] += 1                   # the value of the counter of the trigram pair at the given indices is incremented by 1
P = (N+1).float()                               # the counts matrix is converted to float
P = P / P.sum(dim=2, keepdim=True)              # the probability of each bigram pair is calculated by normalizing with the sum over the row (which are all possible bigram pairs starting with char1)

# generate predictions with the model
preds = []
print("\nGenerating predictions with the basic bigram model...\n")
for i in range(5):
    out = []                                    # this list holds the generated characters of the output name
    ix1 = 0                                     # this is the index of the first and second character (from itos: 0 : "." -> special token to start the sequence)
    ix2 = 0
    while True:
        p = P[ix1, ix2]                         # get the probabilities of all possbible following characters
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()  # sample index according to the probabilities
        out.append(itos[ix])                    # convert index to character and store character
        if ix == 0:                             # if the characters is the special token, end the sequence
            break
    pred = "".join(out)                         # join all chars to get the final predition
    preds.append(pred)

# evaluate the predictions
log_likelihood = 0.0                            # init metric and counter for seen samples
n = 0
for i, pred in enumerate(preds):                # iterate over all generated predictions
    for char1, char2, char3 in zip(pred, pred[1:], pred[2:]):
        ix1 = stoi[char1]
        ix2 = stoi[char2]
        ix3 = stoi[char3]

        prob = P[ix1, ix2, ix3]                 # get the probability of the generated trigram
        log_prob = torch.log(prob)              # calculate the logarithmic probability of the generated tigram pair

        log_likelihood += log_prob              # sum up the logarithmic probabilities of all trigrams of the generated prediction to get the log liklihood (equal to log of product of the single probabilities) 
        neg_log_likelihood = -log_likelihood    # invert the logarithmic likelihood (will be a positive value with range [0, inf])

        n += 1
    avg_log_likelihood = neg_log_likelihood / n # calculate the average logarithmic likelihood over all trigrams of the prediction (can be seen as a loss value)
    print(f"prediction: '{pred}'")
    print(f"Average log liklihood: {avg_log_likelihood:.4f}\n")

# evaluate all samples from the dataset (same as before, just over whole dataset)
log_likelihood = 0.0
n = 0
for word in words:
    for char1, char2, char3 in zip(word, word[1:], word[2:]):
        ix1 = stoi[char1]
        ix2 = stoi[char2]
        ix3 = stoi[char3]

        prob = P[ix1, ix2, ix3]
        log_prob = torch.log(prob)

        log_likelihood += log_prob
        neg_log_likelihood = -log_likelihood

        n += 1
avg_log_likelihood = neg_log_likelihood / n
print(f"Average log liklihood over whole dataset: {avg_log_likelihood:.4f}")