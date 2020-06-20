# cipher
A substitution cipher breaker for the 6.437 final project.

Given a ciphertext encoded with an unknown substitution cipher, we use MCMC with bigram log likelihood to sample the MAP cipher and decode the ciphertext.

We also extend the code for ciphertexts encoded with two different substitution ciphers (the first is used until some unknown breakpoint location in the plaintext, then the second is used for the rest of the plaintext).

`reports/` contain a more complete writeup of the problem, methods, and results.
`demo.ipynb` contins a quick demo.

# Existing optimizations

The primary optimizations used are 1) chain ensembles and 2) accelerated likelihoods calculations. 
1. The ensemble size (and chain length) are configurable at the top of `decode.py`. Increasing the number of chains decreases the probability of missing the MAP cipher at the cost of time. 
2. The likelihood calculation is accelerated through careful of bigram frequencies. A naive likelihood calculation determines how closely (for a given cipher) the decoded text matches typical bigram frequencies in the English language. Since this only depends on bigram frequencies (and not order), we can quickly find log likelihoods by reindexing our frequency counts rather than decoding and recounting bigrams for each newly proposed cipher. This speedup allows us to increase ensemble size thereby increasing the probability of finding the MAP cipher.

# Future improvements

The main remaining hurdle is that the bigram model for the English language is not ideal. For example, the decoded string "qust" may have a higher log likelihood than "just" because the bigram "qu" has a higher likelihood than "ju". This means even if we find the maximum a posteriori estimator, it may not be the best cipher under the English language. One way to help fix this could be to integrate a trigram model into the MCMC.
