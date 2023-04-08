from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse
import spacy
from spacy.tokens import Doc

class SpacyParser(Parser):

    def __init__(self, model_name: str):
        self.model = spacy.load(model_name)

    def parse(self,sentence: str, tokens: list): #-> DependencyParse:
        """Use the specified spaCy model to parse the sentence.py.

        The function should return the parse in the Dependency format.

        You should refer to the spaCy documentation for how to use the model for dependency parsing.
        """
        custom_tokenize = Doc(self.model.vocab,tokens)
        spacy_op = self.model(custom_tokenize)
        heads = []
        deps = []
        for token in spacy_op:
            head_id = tokens.index(token.head.text)+1
            token_dep = token.dep_
            if token_dep=="ROOT":
                token_dep = "root"
                head_id = 0
            deps.append(token_dep)
            heads.append(str(head_id))
        return DependencyParse(sentence,tokens,heads,deps)
