import numpy as np
from collections import defaultdict


NEG_INFINITY = -20

class NGramVanilla(object):
    def __init__(self, n, vsize):
        self.n = n
        self.count = defaultdict(lambda: defaultdict(float))
        self.total = defaultdict(float)
        self.vsize = vsize
    
    def estimate(self, sequences):
        """Estimate the n-gram probabilities up to order self.n.
        
        Specifically, this function updates self.count and self.total as follows:
          1. self.count[prefix][token] should be the number of occurences of prefix followed by token in sequences.
             The special tokens "<bos>" can occur in the prefix to represent "beginning of sequence". "<eos>" can occur in the  
          2. self.total[prefix] is the total number of occurences of prefix in sequences.
        
        Args:
          sequences: A list of lists, each of which represents a sentence (list of words).
          
        Example:
          Arguments:
            self.n = 2
            sequences = [["hello world"], ["hello, "there"]]
          After running:
            self.counts["<bos>", "hello"]["world"] = 1
            self.total["<bos>", "hello"] = 2
        """
        # TODO: Your code here!
        for seq in sequences:
          tokens = ['<bos>']*(self.n-1) + seq + ['<eos>']
          for i in range(len(tokens) - self.n+1):
            ngram = tuple(tokens[i:i+self.n])
            prefix = ngram[:-1]
            word = ngram[-1]
            if self.n == 1 and word == '<eos>':
              continue
            self.total[prefix] += 1
            self.count[prefix][word] += 1
        # End of your code.

    def ngram_prob(self, ngram):
        """Return the probability of the n-gram estimated by the model."""
        prefix = ngram[:-1]
        word = ngram[-1]
        if self.total[prefix] == 0:
          return 0
        return self.count[prefix][word] / self.total[prefix]

    def sequence_logp(self, sequence):
        padded_sequence = ['<bos>']*(self.n-1) + sequence + ['<eos>']
        total_logp = 0
        for i in range(len(padded_sequence) - self.n+1):
            ngram = tuple(padded_sequence[i:i+self.n])
            p = self.ngram_prob(ngram)
            total_logp += max(np.log2(p), NEG_INFINITY)
        return total_logp
