import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
print(f"Using device: {device}")

import string
import unicodedata

# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in allowed_characters
    )


print(f"converting 'Ślusàrski' to {unicodeToAscii('Ślusàrski')}")


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# notice that the first position in the tensor = 1
print(f"The letter 'a' becomes {lineToTensor('a')}")
# notice 'A' sets the 27th index to 1
print(f"The name 'Ahn' becomes {lineToTensor('Ahn')}")
