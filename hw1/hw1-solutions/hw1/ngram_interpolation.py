import numpy as np
from collections import defaultdict

from ngram_vanilla import NGramVanilla, NEG_INFINITY


class NGramInterpolation(NGramVanilla):

    def __init__(self, lambdas, vsize):
        self.lambdas = lambdas
        self.sub_models = [NGramVanilla(n, vsize) for n in range(1, len(lambdas) + 1)]
    
    def estimate(self, sequences):
        for sub_model in self.sub_models:
            sub_model.estimate(sequences)

    def ngram_prob(self, ngram):
        """Return the smoothed probability of an n-gram with interpolation smoothing.
        
        Hint: Call ngram_prob on each vanilla n-gram model in self.sub_models!
        """
        # TODO: Your code here!
        p = 0
        for lamb, sub_model in zip(self.lambdas, self.sub_models):
            p += lamb * sub_model.ngram_prob(ngram[-sub_model.n:])
        return p
        # End of your code.

    def sequence_logp(self, sequence):
        padded_sequence = ['<bos>']*(len(self.lambdas)-1) + sequence + ['<eos>']
        total_logp = 0
        for i in range(len(padded_sequence) - len(self.lambdas)+1):
            ngram = tuple(padded_sequence[i:i+len(self.lambdas)])
            p = self.ngram_prob(ngram)
            total_logp += max(np.log2(p), NEG_INFINITY)
        return total_logp
