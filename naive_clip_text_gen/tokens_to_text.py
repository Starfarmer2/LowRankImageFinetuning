# tokens = ['', 'k', 'ups</w>', 'oficial</w>', 'mein</w>', 'dazz', 'roofing</w>', 'super', 'du</w>', 'swach', 'counter</w>', 'standup</w>', 'Ĥâĸ', 'printing</w>', 'irresistible</w>', 'dispen', 'tipoff</w>', 'mediocre</w>', 'guan</w>', 'icious</w>', 'attach', 'slush</w>', 'slush</w>', 'cavity</w>', 'ntv</w>', 'isner</w>', 'dryer</w>', 'hoc</w>', 'abscbn</w>', 'dep', 'ffy</w>', 'bara</w>', '']
# tokens = ['<|startoftext|>', 'impact</w>', 'collapse</w>', 'pillow</w>', 'sheldon</w>', 'delia</w>', 'sheldon</w>', 'nuestra</w>', 'ðŁĶĿ', 'accordion</w>', 'counter</w>', 'dim</w>', 'pillow</w>', 'dry', 'pillow</w>', 'spel', 'sleeves</w>', 'reward</w>', 'pillow</w>', 'pouch</w>', 'coordination</w>', 'autu', 'lourdes</w>', 'chia</w>', 'canteen</w>', 'sain</w>', 'machining</w>', 'gical</w>', 'firms</w>', 'flint</w>', 'cy</w>', 'bara</w>', '<|endoftext|>']
# tokens = ['<|startoftext|>', 'impact</w>', 'collapse</w>', 'pillow</w>', 'sheldon</w>', 'delia</w>', 'sheldon</w>', 'nuestra</w>', 'ðŁĶĿ', 'accordion</w>', 'counter</w>', 'dim</w>', 'pillow</w>', 'dry', 'pillow</w>', 'spel', 'sleeves</w>', 'reward</w>', 'pillow</w>', 'pouch</w>', 'coordination</w>', 'autu', 'lourdes</w>', 'chia</w>', 'canteen</w>', 'sain</w>', 'machining</w>', 'gical</w>', 'firms</w>', 'flint</w>', 'cy</w>', 'bara</w>', '<|endoftext|>']
# tokens = ['<|startoftext|>', 'bag</w>', 'tre</w>', 'eera</w>', 'toid</w>', 'doh</w>', 'emmy</w>', 'ization</w>', 'signature</w>', 'rahman</w>', '<|endoftext|>']
# tokens = [porsche kato mpmillionaire xclap cheerleading]
# tokens = ['<|startoftext|>', 'porsche</w>', 'fanatic</w>', 'punch', 'bond</w>', 'worsen', 'pulp</w>', 'presser</w>', 'media</w>', 'hornet</w>', 'fez</w>', 'for', 'ire', 'tea</w>', 'reformed</w>', 'rhodes</w>', 'for', 'mp', 'presser</w>', 'pembroke', 'persi', 'mber', 'sery</w>', 'presser</w>', 'jillian</w>', 'gerry', 'mp', 'kran', 'compass</w>', 'ew', 'ire', 'camber', 'attor', 'titans</w>', 'gerry', 'doubled</w>', 'pitch</w>', 'digger</w>', 'sorry</w>', 'united</w>', 'canned</w>', 'recruiter</w>', 'sltd</w>', 'hangar</w>', 'tened</w>', 'scot</w>', '<|endoftext|>']
# tokens = ['<|startoftext|>', 'firms</w>', 'fez</w>', 'firms</w>', 'firms</w>', 'mug</w>', 'airport</w>', 'firms</w>', 'club</w>', 'donations</w>', 'heads</w>', 'shuts</w>', 'fez</w>', 'firms</w>', 'mug</w>', 'chilly</w>', 'fab</w>', 'bit</w>', 'copied</w>', 'economics</w>', 'wba</w>', 'fez</w>', 'ghe</w>', 'fez</w>', 'sz</w>', 'fez</w>', 'fez</w>', 'kojima</w>', 'ledge</w>', 'ðŁĮ¸</w>', 'jock</w>', 'bankers</w>', 'Ľ</w>', 'wire</w>', 'amu</w>', 'sns</w>', 'working</w>', 'deephouse</w>', 'arm</w>', 'rowing</w>', 'fez</w>', 'stardom</w>', 'heads</w>', 'fez</w>', 'fez</w>', 'faint</w>', '<|endoftext|>']
tokens = ['<|startoftext|>', 'bjj</w>', 'charitable</w>', 'sarawak</w>', 'dl</w>', 'ffler</w>', 'sis</w>', 'sheldon</w>', 'pouch</w>', 'ded</w>', 'pouch</w>', 'bjj</w>', 'ization</w>', 'ded</w>', 'sheldon</w>', 'rebs</w>', 'billboard</w>', 'ded</w>', 'sleeves</w>', 'sleeves</w>', 'pillow</w>', 'qaeda</w>', 'delia</w>', 'sarawak</w>', 'ppm</w>', 'hedgehog</w>', 'gui</w>', 'delia</w>', 'sleeves</w>', 'sleeves</w>', 'pillow</w>', 'delia</w>', 'delia</w>', 'dees</w>', 'hns</w>', 'workers</w>', 'delia</w>', 'delia</w>', 'peeling</w>', 'basket</w>', 'cleats</w>', 'sleeves</w>', 'ups</w>', 'gui</w>', 'rebs</w>', 'tress</w>', 'loh</w>', 'roofing</w>', 'caddy</w>', 'blazer</w>', 'lender</w>', '<|endoftext|>']

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