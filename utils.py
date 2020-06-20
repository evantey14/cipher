import numpy as np

def o(c): # a better version of ord
    if c == ' ':
        return 26
    elif c == '.':
        return 27
    else:
        return ord(c) - 97
    
def c(o): # a better version of chr
    alphabet = 'abcdefghijklmnopqrstuvwxyz .'
    return alphabet[o]

def text_to_number(text):
    return np.array([o(letter) for letter in text])

def number_to_text(nums):
    return ''.join([c(num) for num in nums])

def get_accuracy(original, guess, is_text=False):
    if is_text:
        return get_accuracy(text_to_number(original), text_to_number(guess))
    return np.count_nonzero(original == guess) / len(original)

def decode_with_f(f, text):
    return f[text]

def decode_with_b(f1, f2, b, text):
    text1 = f1[text[:b+1]]
    text2 = f2[text[b+1:]]
    return np.concatenate([text1, text2])

def count_bigrams(cipher):
    bigram_counts = np.zeros((28, 28))
    for i in range(len(cipher) - 1):
        bigram_counts[cipher[i+1]][cipher[i]] += 1
    return bigram_counts

def count_bigrams_over_time(cipher):
    bigram_counts = np.zeros((len(cipher), 28, 28))
    for i in range(len(cipher) - 1):
        bigram_counts[i+1][cipher[i+1]][cipher[i]] += 1
        bigram_counts[i+1] += bigram_counts[i]
    return bigram_counts

def permute_bigram_counts(f, bigram_counts):
    '''Permute bigram counts under permutation f.'''
    f_inv = np.argsort(f)
    return bigram_counts[f_inv, :][:, f_inv]

def sample_f_neighbor(f):
    f_prime = np.copy(f)
    i1, i2 = np.random.choice(28, 2)
    f_prime[i1], f_prime[i2] = f_prime[i2], f_prime[i1]
    return f_prime

def sample_cipher_neighbor(f1, f2, b, ciphertext_length):
    f1_prime = sample_f_neighbor(f1)
    f2_prime = sample_f_neighbor(f2)
    b_prime = (np.random.normal(loc=b, scale=21) % ciphertext_length).astype(np.int)
    return f1_prime, f2_prime, b_prime

def sample_f():
    return np.random.permutation(28)

def sample_b(length):
    return np.random.randint(length)

def get_random_substring(text):
    start = np.random.randint(len(text))
    length = np.random.randint(256, 2048) # length bounds defined in handout
    end = min(start + length, len(text))
    return text[start:end]
