import torch 
import matplotlib.pyplot as plt


# load all the names from the input file, split at each line and return all elements in a single list
words = open("names.txt", "r").read().splitlines()

# print the first 10 names of the list
print("\nFirst 10 names of the input data:")
print(words[:10])

# print the shortest name
print("\nShortest name of the input data:")
print(min(len(word) for word in words))

# print the longest name
print("\nLongest name of the input data:")
print(max(len(word) for word in words))

# print a heatmap of the bigrams which were found
heatmap = torch.zeros((28, 28), dtype=torch.int32)

chars = sorted(list(set("".join(words))))

stoi = {s:i for i,s in enumerate(chars)}
stoi["<S>"] = 26
stoi["<E>"] = 27

for word in words:
    chars = ["<S>"] + list(word) + ["<E>"]
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        heatmap[ix1, ix2] += 1

itos = {i:s for s,i in stoi.items()}

plt.imshow(heatmap)
plt.waitforbuttonpress()
