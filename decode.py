import numpy as np

from utils import *

np.seterr(all='ignore') # test server requires silent stdout

# Configure the MCMC
STOP_THRESHOLD = 1000
MAX_ITERS = 2500
ENSEMBLE_SIZE = 50

# Load bigram log probabilities and store as a 1D array, adding noise to zero transitions
BIGRAM_PS = np.loadtxt('data/letter_transition_matrix.csv', delimiter=',')
BIGRAM_LOGPS = np.log(BIGRAM_PS + 1e-8).ravel()


def decode(ciphertext, has_breakpoint, debug=False):
    ciphernumbers = text_to_number(ciphertext) # convert to number list for computations

    if has_breakpoint:
        bigram_counts_over_time = count_bigrams_over_time(ciphernumbers)
        maxes, histories = (None, None, None, -np.inf), [] # (f1, f2, b, logp), history arrays
        gen_chain = lambda: generate_chain_with_breakpoint(bigram_counts_over_time)
    else:
        bigram_counts = count_bigrams(ciphernumbers)
        maxes, histories = (None, -np.inf), [] # (f, logp), history arrays
        gen_chain = lambda: generate_chain(bigram_counts)

    for _ in range(ENSEMBLE_SIZE):
        chain_max, chain_history = gen_chain()
        histories.append(chain_history)
        if chain_max[-1] > maxes[-1]:
            maxes = chain_max

    if has_breakpoint:
        f1_hat, f2_hat, b_hat = maxes[0], maxes[1], maxes[2]
        plainnumbers = decode_with_b(f1_hat, f2_hat, b_hat, ciphernumbers)
    else:
        f_hat = maxes[0]
        plainnumbers = decode_with_f(f_hat, ciphernumbers)

    return number_to_text(plainnumbers)

def generate_chain_with_breakpoint(bigram_counts_over_time):
    # here, a cipher consists of a triple (f1, f2, b)
    bigram_counts = bigram_counts_over_time[-1]
    ciphertext_length = (np.sum(bigram_counts) + 1).astype(np.int)
    f1s, f2s, bs = initialize_ciphers(bigram_counts, ciphertext_length)
    logp0 = log_likelihood_with_breakpoint(f1s[0], f2s[0], bs[0], bigram_counts_over_time)
    logps = initialize_logps(logp0)
    maxes = f1s[0], f2s[0], bs[0], logps[0]
    for i in range(1, MAX_ITERS):
        if is_stuck(i, logps):
            break
        cipher_old = f1s[i-1], f2s[i-1], bs[i-1]
        cipher_new = sample_cipher_neighbor(*cipher_old, ciphertext_length)

        logp_old = logps[i-1]
        logp_new = log_likelihood_with_breakpoint(*cipher_new, bigram_counts_over_time)
        acceptance_rate = min(1, np.exp(logp_new - logp_old))
        accept_proposal = np.random.uniform(1) <= acceptance_rate

        f1s[i], f2s[i], bs[i] = cipher_new if accept_proposal else cipher_old
        logps[i] = logp_new if accept_proposal else logp_old
        if logps[i] > maxes[-1]:
            maxes = *cipher_new, logp_new
    return maxes, (f1s, f2s, bs, logps)

def generate_chain(bigram_counts):
    fs = initialize_fs(bigram_counts)
    logps = initialize_logps(log_likelihood(fs[0], bigram_counts))
    maxes = fs[0], logps[0]
    for i in range(1, MAX_ITERS):
        if is_stuck(i, logps):
            break
        f_old, f_new = fs[i-1], sample_f_neighbor(fs[i-1])
        logp_old, logp_new = logps[i-1], log_likelihood(f_new, bigram_counts)
        acceptance_rate = min(1, np.exp(logp_new - logp_old))
        accept_proposal = np.random.uniform(1) <= acceptance_rate

        fs[i], logps[i] = (f_new, logp_new) if accept_proposal else (f_old, logp_old)
        if logps[i] > maxes[-1]:
                maxes = f_new, logp_new
    return maxes, (fs, logps)

def initialize_fs(bigram_counts):
    fs = np.zeros((MAX_ITERS, 28), dtype=np.int)
    fs[0] = sample_f()
    return fs

def initialize_ciphers(bigram_counts, ciphertext_length):
    f1s = initialize_fs(bigram_counts)
    f2s = initialize_fs(bigram_counts)
    bs = np.zeros(MAX_ITERS, dtype=np.int)
    bs[0] = np.random.randint(ciphertext_length)
    return f1s, f2s, bs

def initialize_logps(logps0):
    logps = np.ones(MAX_ITERS) * np.nan
    logps[0] = logps0
    return logps

def log_likelihood(f, bigram_counts):
    """Calculate the log likelihood of ciphertext with a set of bigram counts under cipher f.

    Args:
        f: length-28 array, representation of substitution cipher.
        bigram_counts: 28x28 array, bigram frequencies in ciphertext.
    """
    new_bigram_counts = permute_bigram_counts(f, bigram_counts).ravel()
    loglikelihood = new_bigram_counts.dot(BIGRAM_LOGPS)
    return loglikelihood

def log_likelihood_with_breakpoint(f1, f2, b, bigram_counts):
    bigram_counts_before = bigram_counts[b]
    log_likelihood1 = log_likelihood(f1, bigram_counts_before)

    bigram_counts_after = bigram_counts[-1] - bigram_counts_before
    log_likelihood2 = log_likelihood(f2, bigram_counts_after)

    return log_likelihood1 + log_likelihood2

def is_stuck(i, logps):
    return i > STOP_THRESHOLD and logps[i-1] == logps[i-1 - STOP_THRESHOLD]
