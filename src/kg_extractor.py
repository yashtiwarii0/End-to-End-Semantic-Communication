# src/kg_extractor.py

import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")


def clean_phrase(tokens):
    """
    Remove unnecessary words like:
    a, an, the
    """

    remove_words = {"a", "an", "the"}

    cleaned = []

    for token in tokens:
        if token.text.lower() not in remove_words:
            cleaned.append(token.text)

    return " ".join(cleaned)


def extract_triples(sentence):

    doc = nlp(sentence)

    subject = None
    relation = None
    obj = None

    for token in doc:

        # Subject
        if token.dep_ == "nsubj":
            subject = clean_phrase(token.subtree)

        # Main verb
        elif token.dep_ == "ROOT":
            relation = token.text

        # Direct object
        elif token.dep_ == "dobj":
            obj = clean_phrase(token.subtree)

    if subject and relation and obj:
        return [(subject, relation, obj)]

    return []