from kg_extractor import extract_triples
from entropy_checker import calculate_entropy
from kg_extractor import extract_triples
from entropy_checker import calculate_entropy
from encoder import triples_to_text, encode_text

THRESHOLD = 3.85

sentence = input("Enter sentence: ")

entropy = calculate_entropy(sentence)

print("Entropy:", entropy)

if entropy > THRESHOLD:

    print("Unstructured Sentence")
    print("Skipping KG Extraction")

else:

    triples = extract_triples(sentence)

    if triples:
        print("Structured Sentence")
        print("Triples:", triples)

        # Convert triples to text
        text = triples_to_text(triples)

        print("\nText for T5:")
        print(text)

    # Encode with T5
        embedding = encode_text(text)

        print("\nEmbedding Shape:")
        print(embedding.shape)

    else:
        print("No valid KG triples found")
        print("Treating as unstructured sentence")



#sentence = "Virat Kohli scored a century against Australia."
