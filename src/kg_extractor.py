import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")


def clean_phrase(tokens):
    """
    Remove unnecessary words:
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

    triples = []

    # First pass: find subject, main verb, direct object
    for token in doc:

        if token.dep_ == "nsubj":
            subject = clean_phrase(token.subtree)

        elif token.dep_ == "ROOT":
            relation = token.text

        elif token.dep_ in ("dobj", "obj", "attr","npadvmod"):
            obj = clean_phrase(token.subtree)
    
    # Fallback subject detection

    if subject is None:

        first_token = doc[0]
        if first_token.text.lower() in ("the", "a", "an"):
            
            for token in doc:
                if token.pos_ in ("PROPN", "NOUN"):
                    subject = token.text
                    break
        else:
            subject = first_token.text
    # Main triple
    if subject and relation and obj:
        triples.append((subject, relation, obj))

    # Second pass: find preposition relations
    for token in doc:

        if token.dep_ == "prep":

            prep_relation = token.text

            for child in token.children:

                if child.dep_ == "pobj":

                    prep_object = clean_phrase(child.subtree)

                    triples.append(
                        (subject, prep_relation, prep_object)
                    )

    return triples