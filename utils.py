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

def get_accuracy(original, guess):
    return np.count_nonzero(original == guess) / len(original)

def decode_with_f(f, text):
    return f[text]

def decode_with_b(f1, f2, b, text):
    text1 = f1[text[:b+1]]
    text2 = f2[text[b+1:]]
    return np.concatenate([text1, text2])

def text_to_number(text):
    return np.array([o(letter) for letter in text])

def number_to_text(nums):
    return ''.join([c(num) for num in nums])
