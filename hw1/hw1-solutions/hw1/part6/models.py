# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        #return [self.predict(ex_words) for ex_words in all_ex_words]
        return self.predict(all_ex_words)


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class NeuralNet(nn.Module):

    def __init__(self, word_embeddings, hidden_dim):
        super().__init__()
        self.word_embeddings = word_embeddings 
        self.hidden_dim = hidden_dim
        # TODO: Your code here!
        self.embedding_dim = self.word_embeddings.get_embedding_length()
        self.fc1 = nn.Linear(self.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2) 
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, ex_words, token_len) -> torch.Tensor:
        """Takes a list of words and returns a tensor of class predictions.

        Steps:
          1. Embed each word using self.embeddings.
          2. Take the average embedding of each word.
          3. Pass the average embedding through a neural net with one hidden layer (with ReLU nonlinearity).
          4. Return a tensor representing a log-probability distribution over classes.
        """
        # TODO: Your code here!

        output = torch.div(torch.sum(ex_words, dim=1), token_len.unsqueeze(1))
        
        output = self.relu(self.fc1(output))
        output = self.fc2(output)

        return F.log_softmax(output, dim=1)

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, word_embeddings, hidden_dim): # TODO originally __init__(self)
        # TODO: Your code here!
        self.hidden_dim = hidden_dim
        self.word_embeddings = word_embeddings 
        self.net = NeuralNet(self.word_embeddings, hidden_dim) 

    def predict(self, ex_words: List[str]) -> int:
        # TODO: Your code here! 
        embeddings = []
        token_len = []
        for exs in ex_words:
            embeddings.append(torch.Tensor([self.word_embeddings.get_embedding(w) for w in exs]))
            token_len.append(len(exs))
        embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        token_len = torch.Tensor(token_len)
        logits = self.net(embeddings, token_len)
        return torch.argmax(logits, dim=1)

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # TODO: Your code here!
    classifier = NeuralSentimentClassifier(word_embeddings, args.hidden_size) 
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(classifier.net.parameters(), lr=args.lr)

    embeddings_train = []
    token_len_train = []
    y_train = []
    for exs in train_exs:
        embeddings_train.append(torch.Tensor([word_embeddings.get_embedding(w) for w in exs.words]))
        token_len_train.append(len(exs.words))
        y_train.append(exs.label)

    embeddings_train = nn.utils.rnn.pad_sequence(embeddings_train, batch_first=True)
    train_data = [[x,y,l] for x,y,l in zip(embeddings_train, y_train, token_len_train)]


    embeddings_dev = []
    token_len_dev = []
    y_dev = []
    for exs in dev_exs:
        embeddings_dev.append(torch.Tensor([word_embeddings.get_embedding(w) for w in exs.words]))
        token_len_dev.append(len(exs.words))
        y_dev.append(exs.label)
    embeddings_dev = nn.utils.rnn.pad_sequence(embeddings_dev, batch_first=True)
    embeddings_dev = torch.Tensor(embeddings_dev)
    token_len_dev = torch.Tensor(token_len_dev)
    y_dev = torch.LongTensor(y_dev)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.num_epochs):
        eval_loss = []
        for _,batch in enumerate(trainloader):
            x, y, l = batch
            optimizer.zero_grad()
            classifier.net.train()
            y_hat = classifier.net(x,l) 
            
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            classifier.net.eval()

            with torch.no_grad():
                dev_pred = classifier.net(embeddings_dev, token_len_dev)
                eval_loss.append(criterion(dev_pred, y_dev))

        print("EPOCH: %d Loss: %.4f" % (epoch, np.mean(eval_loss)))
    
    return classifier

