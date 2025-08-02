from tokenizer import tokenize, encode, decode, build_vocab

merges, max_index, all_indices, ids, vocab_size = tokenize()  # modifica tokenize() per restituire merges
tokens = encode("hi my name is andrea \nand i'm an animal", merges)
print(tokens)

vocab = build_vocab(merges, max_index)

decoded = decode(tokens, vocab)
print(decoded)