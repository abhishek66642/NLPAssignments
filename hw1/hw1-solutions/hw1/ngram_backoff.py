import numpy as np
from collections import defaultdict

from ngram_vanilla import NGramVanilla


class NGramBackoff(NGramVanilla):
    def __init__(self, n, vsize):
        self.n = n
        self.count = defaultdict(lambda: defaultdict(float))
        self.total = defaultdict(float)
        self.vsize = vsize
        self.sub_models = [NGramVanilla(n, vsize) for n in range(1, self.n + 1)]
    
    def estimate(self, sequences):
        for sub_model in self.sub_models:
            sub_model.estimate(sequences)

    def ngram_prob(self, ngram):
        """Return the smoothed probability with backoff.
        
        That is, if the n-gram count of size self.n is defined, return that.
        Otherwise, check the n-gram of size self.n - 1, self.n - 2, etc. until you find one that is defined.
        
        Hint: Refer to ngram_prob in ngrams_vanilla.py.
        """
        # TODO: Your code here!
        for i in range(self.n):
            submodel = self.sub_models[-i-1]
            token = ngram[i:]
            prefix = token[:-1]
            word = token[-1]
            if submodel.count[prefix][word] > 0:
                return submodel.ngram_prob(token)
        return 0
        # End of your code.
