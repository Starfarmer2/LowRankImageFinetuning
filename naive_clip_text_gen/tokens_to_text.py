# tokens = ['', 'k', 'ups</w>', 'oficial</w>', 'mein</w>', 'dazz', 'roofing</w>', 'super', 'du</w>', 'swach', 'counter</w>', 'standup</w>', 'Ĥâĸ', 'printing</w>', 'irresistible</w>', 'dispen', 'tipoff</w>', 'mediocre</w>', 'guan</w>', 'icious</w>', 'attach', 'slush</w>', 'slush</w>', 'cavity</w>', 'ntv</w>', 'isner</w>', 'dryer</w>', 'hoc</w>', 'abscbn</w>', 'dep', 'ffy</w>', 'bara</w>', '']
# tokens = ['<|startoftext|>', 'impact</w>', 'collapse</w>', 'pillow</w>', 'sheldon</w>', 'delia</w>', 'sheldon</w>', 'nuestra</w>', 'ðŁĶĿ', 'accordion</w>', 'counter</w>', 'dim</w>', 'pillow</w>', 'dry', 'pillow</w>', 'spel', 'sleeves</w>', 'reward</w>', 'pillow</w>', 'pouch</w>', 'coordination</w>', 'autu', 'lourdes</w>', 'chia</w>', 'canteen</w>', 'sain</w>', 'machining</w>', 'gical</w>', 'firms</w>', 'flint</w>', 'cy</w>', 'bara</w>', '<|endoftext|>']
tokens = ['<|startoftext|>', 'impact</w>', 'collapse</w>', 'pillow</w>', 'sheldon</w>', 'delia</w>', 'sheldon</w>', 'nuestra</w>', 'ðŁĶĿ', 'accordion</w>', 'counter</w>', 'dim</w>', 'pillow</w>', 'dry', 'pillow</w>', 'spel', 'sleeves</w>', 'reward</w>', 'pillow</w>', 'pouch</w>', 'coordination</w>', 'autu', 'lourdes</w>', 'chia</w>', 'canteen</w>', 'sain</w>', 'machining</w>', 'gical</w>', 'firms</w>', 'flint</w>', 'cy</w>', 'bara</w>', '<|endoftext|>']

def reconstruct_text(tokens):
    # Initialize an empty list to hold words
    words = []
    current_word = ''
    
    # Iterate over each token in the list
    for token in tokens:
        if token.endswith('<|startoftext|>') or token.endswith('<|endoftext|>'):
            continue
        if token.endswith('</w>'):
            # If the token ends with '</w>', it is the end of a word
            current_word += token[:-4]  # Remove '</w>' and add to current word
            words.append(current_word)  # Add the complete word to the list
            current_word = ''  # Reset current word
        else:
            # If not ending in '</w>', continue forming the current word
            current_word += token
    
    # Join all words with spaces to form the reconstructed text
    return ' '.join(words)

reconstructed_text = reconstruct_text(tokens)
print(reconstructed_text)