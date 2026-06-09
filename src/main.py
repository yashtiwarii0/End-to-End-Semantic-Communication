from kg_extractor import extract_triples
from entropy_checker import calculate_entropy

THRESHOLD = 3.85

sentence = input("Enter sentence: ")

entropy = calculate_entropy(sentence)

print("Entropy:", entropy)

if entropy > THRESHOLD:

    print("Unstructured Sentence")
    print("Skipping KG Extraction")

else:

    triples = extract_triples(sentence)

    print("Structured Sentence")
    print("Triples:", triples)



#sentence = "Virat Kohli scored a century against Australia."
