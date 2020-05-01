"""
Script for generating ciphertexts.
Usage: python3 encode.py plaintext.out ciphertext.out has_breakpoint [seed]

Behavior:
    1. Reads in standard input (until EOF).
    2. Cleans text to satisfy requirements given in the project handout.
    3. Writes the cleaned text to `plaintext.out`.
    4. Encodes the cleaned text and writes the ciphertext to `ciphertext.out`.

Setting has_breakpoint to true encodes with a breakpoint.
Passing a seed as the optional last argument makes the encoding deterministic.

Example invocations:
    python3 encode.py plaintext.txt ciphertext.txt false 42 < data/texts/feynman.txt
    python3 encode.py plaintext.txt ciphertext.txt true < data/texts/tolstoy.txt

Can also be used for cleaning text in the following way:
    python3 encode.py clean.txt /dev/null 0 < dirty.txt
"""

import sys
import string
import random
import unicodedata

ALPHABET = list(string.ascii_lowercase) + [" ", "."]
LETTER_TO_IDX = dict(map(reversed, enumerate(ALPHABET)))


def clean_text(text):
    # try and approximate unicode with ascii
    text = unicodedata.normalize("NFKD", text).encode("ascii",
                                                      "ignore").decode()

    text = text.lower()  # make lowercase
    text = text.replace("?", ".").replace("!", ".")
    for c in "/-\n\r":
        text = text.replace(c, " ")
    text = "".join(filter(ALPHABET.__contains__,
                          text))  # filter to alphabet chars

    text = text.lstrip(" .")  # filter out leading spaces and periods
    if text == "":
        raise ValueError("text needs to have at least one letter")

    ret = ""
    for x in text:
        # ret is a valid string after every iteration
        if x == ".":
            ret = ret.rstrip(". ") + ". "
        elif x == " ":
            ret = ret.rstrip(" ") + " "
        else:
            ret += x

    ret = ret.rstrip(" ")  # strip trailing spaces
    return ret


def encode(plaintext):
    cipherbet = ALPHABET.copy()
    random.shuffle(cipherbet)

    ciphertext = "".join(cipherbet[LETTER_TO_IDX[c]] for c in plaintext)
    return ciphertext


def encode_with_breakpoint(plaintext):
    bpoint = random.randint(0, len(plaintext))
    print(f"Breakpoint at position {bpoint}")

    return encode(plaintext[:bpoint]) + encode(plaintext[bpoint:])


def main():
    plaintext_out = sys.argv[1]
    ciphertext_out = sys.argv[2]
    has_breakpoint = (sys.argv[3].lower() == "true")
    if len(sys.argv) > 4:
        random.seed(sys.argv[4])

    raw_text = sys.stdin.read()

    plaintext = clean_text(raw_text)
    print(f"Clean plaintext length: {len(plaintext)}")
    with open(plaintext_out, "w") as f:
        f.write(plaintext)

    if has_breakpoint:
        print("Encoding with breakpoint...")
        ciphertext = encode_with_breakpoint(plaintext)
    else:
        print("Encoding without breakpoint")
        ciphertext = encode(plaintext)

    with open(ciphertext_out, "w") as f:
        f.write(ciphertext)


if __name__ == "__main__":
    main()
