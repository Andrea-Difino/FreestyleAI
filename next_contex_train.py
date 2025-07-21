string = "Io mangio caviale nero <START>"

wotoi = {"<END>": 0, "<START>": 1, "Io": 2, "mangio": 3, "caviale": 4, "nero": 5}
context_size = 4

X, Y = [], []
context = [wotoi["<END>"]] * context_size
for word in reversed(string.split()):
    ix = wotoi[word]
    X.append(context.copy())
    Y.append(ix)
    context = [ix] + context[:-1]

print(X)
print(Y)