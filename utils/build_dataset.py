import torch


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