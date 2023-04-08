from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse
#from src.bert_parser_model import BertParserModel

import networkx as nx
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence
from transformers import DistilBertTokenizer
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class BertParser(Parser):

    """Represents a full parser that decodes parse trees from a finetuned BERT model."""

    def __init__(self, model_path: str, mst: bool = False):
        """Load your saved finetuned model using torch.load().

        Arguments:
            model_path: Path from which to load saved model.
            mst: Whether to use MST decoding or argmax decoding.
        """
        self.mst = mst
        # TODO: Load your neural net.
        self.model = torch.load(model_path)
        self.mst = mst

    def parse(self,sentence: str, tokens: list) -> DependencyParse:
        """Build a DependencyParse from the output of your loaded finetuned model.

        If self.mst == True, apply MST decoding. Otherwise use argmax decoding.        
        """
        # TODO: Your code here!
        
        if (not self.mst):
            encoded_ip_tensor = tokenizer(sentence, return_tensors="pt", padding=True)
            rel_logits,dep_logits = self.model(encoded_ip_tensor)
            rel_pos = torch.argmax(rel_logits,dim=1)
            deprel = torch.argmax(dep_logits,dim=1)
            print (rel_pos)
            print (deprel)
        
        
        
