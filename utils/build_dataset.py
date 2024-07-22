import torch



def load_data(path):
    """
    Returns a list with all training elements as well as the mapping from int to string.

    Params:
        - path: Relative path to the file which contains all the names (str)
    
    Returns:
        - words: Contains all the names of the dataset (List[str]) 
        - vocab_size: number of unique elements of the input data (int)
        - stoi: string to int mapping (dict[str|int])
        - itos: int to string mapping (dict[int|str])   
    """

    # load data, create mapping between strings and integers
    words = open(path, "r").read().splitlines()                     # store data in list, each name is one string
    unique_chars = sorted(list(set("".join(words))))                # get all unique chars of the data
    vocab_size = len(unique_chars) + 1                              # get the number of unique elements
    stoi = {s:i+1 for i,s in enumerate(unique_chars)}               # create string to int mapping       
    stoi["."] = 0                                                   # add "." string at index 0 (this char symbolizes the beginning and end of a sequence)
    itos = {i:s for s,i in stoi.items()}                            # create int to string mapping

    return words, vocab_size, stoi, itos



def build_dataset(words, stoi, block_size):
    """
    This function creates a dataset consisting of a data tensor and a label tensor from a list of strings.
    Params:
        - words: Contains all names of the dataset (List[str])
        - stoi: Mapping between strings and ints (dict[str|int])
        - block_size: size of the context window (int)
    
    Returns:
        - X: data tensor containing the indices of the names (torch.tensor, shape = [N,3])
        - Y: label tensor containing the indices of the labels (torch.tensor, shape = [N])
    """

    # input data and labels
    X, Y, = [],[]

    # iterate over whole training data
    for word in words:

        context = [0] * block_size          # init context to 0,0,0     

        for char in word + ".":             # iterate over all chars of the current word
            ix = stoi[char]                 
            X.append(context)               # create data and labels
            Y.append(ix)
            context = context[1:] + [ix]    # shift elements of context to the right and add new last character

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y