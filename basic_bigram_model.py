import torch 



# load all strings from names.txt, split at each line and return all strings in a single list
words = open("names.txt", "r").read().splitlines()


# create empty matrix (dims are defined by the number of unique characters within the dataset (alphabet has 26 + 1 special char for start and end of a sequence))
# this matrix will hold the number of occurences of each bigram pair
N = torch.zeros((27, 27), dtype=torch.int32)


# get all unique chars from the dataset, store them in a list and sort them in ascending order
unique_chars = sorted(list(set("".join(words))))           


# generate LUT for string-int and int-string conversion, add a special token for start and end of a sequence
stoi = {s:i+1 for i,s in enumerate(unique_chars)}          
stoi["."] = 0 
itos = {i:s for s,i in stoi.items()}                
 

# iterate over all strings within the dataset
for word in words:
    # add the special token at the beginning and the end of the current string                                  
    chars = ["."] + list(word) + ["."]

    # iterate over all bigram pairs of the current string (e.g: word=".Alex." -> bigrams=[".A", "Al", "le", "ex", "x."])
    # and increment the counter of each bigram pair by 1
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


# cast the matrix to float, then normalize all elements of each row by dividing through the sum over all elements of the row
# this will create a probability distribution of each bigram pair
# keepdim=True return a matrix with the dimensions [27, 1]
P = N.float()
P = P / P.sum(dim=1, keepdim=True)  # always check if matrices broadcastable:   27, 27
                                    #                                           27,  1 -> this dimension gets stretched out over the whole row


# create generator for deterministic behaveiour
g = torch.Generator().manual_seed(111)             


# generate random prediction of the model
for i in range(1):
    # create a list to store the prediction
    out = []

    # create first token (index 0 means it is the special token for starting and ending a sequence)
    ix = 0

    while True:
        # get probabilities of all possible bigrams (first row)
        p = P[ix]

        # sample random element from the given probabilities and convert the element to string
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        # append the string to the list
        out.append(itos[ix])

        # if the index 0 is chosen this means the end of the sequence is reached
        if ix == 0:
            break

    pred = "".join(out)

    log_likelihood = 0.0
    n = 0
    for ch1, ch2 in zip(pred, pred[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        neg_log_likelihood = -log_likelihood

        n += 1

        print(f"{ch1}{ch2}: {prob:.4f} {log_prob:.4f}")
        
    
    avg_log_likelihood = log_likelihood / N

