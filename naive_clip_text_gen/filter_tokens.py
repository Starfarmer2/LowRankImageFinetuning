from transformers import CLIPTokenizer

# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


# Get the vocabulary
vocab = tokenizer.get_vocab()

print(list(vocab.values())[:10])  # Print all token indices

# Filter tokens that end with '</w>'
tokens_with_w = {token: idx for token, idx in vocab.items() if token.endswith('</w>')}
print(list(tokens_with_w.values()))
# Print some of the tokens to verify
# print(list(tokens_with_w.items())[:])  # Print first 10 tokens that end with '</w>'
# print(tokens_with_w.values())  # Print first 10 tokens that end with '</w>'
