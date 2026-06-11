from entropy_checker import calculate_entropy
from kg_extractor import extract_triples
from encoder import triples_to_text, encode_text
from similarity import calculate_similarity
import csv
from channel import add_awgn_noise
THRESHOLD = 3.85
def load_sentences(filepath):

    with open(filepath, "r", encoding="utf-8") as file:

        sentences = []

        for line in file:

            line = line.strip()

            if line:
                sentences.append(line)

        return sentences

sentences = load_sentences("../data/sample_sentences.txt")

with open("../results.csv", "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Sentence",
        "Entropy",
        "Triples",
        "KG_Text",
        "Similarity"
    ])

    for sentence in sentences:
        entropy = calculate_entropy(sentence)

        if entropy > THRESHOLD:
            print("\n" + "=" * 60)
            print("Sentence:")
            print(sentence)

            print("\nEntropy:", entropy)
            print("Skipped (High Entropy)")
            writer.writerow([
            sentence,
            round(entropy, 4),
            "SKIPPED",
            "",
            ""
            ])
            continue

        triples = extract_triples(sentence)

        if not triples:
            print("\n" + "=" * 60)
            print("Sentence:")
            print(sentence)

            print("\nNo valid KG triples found")
            writer.writerow([
            sentence,
            round(entropy, 4),
            "NO_TRIPLES",
            "",
            ""
            ])
            continue

        text = triples_to_text(triples)

        similarity = calculate_similarity(
            sentence,
            text
        )

        writer.writerow([
        sentence,
        round(entropy, 4),
        str(triples),
        text,
        round(similarity, 4)
        ])
        encoded_data = encode_text(text)
        original_embedding = (
        encoded_data["encoder_outputs"]
            .last_hidden_state
        )

        snr_db = 2
        noisy_embedding = add_awgn_noise(
            original_embedding,
            snr_db
        )
        print("\nSNR:")
        print(f"{snr_db} dB")

        print("\nOriginal Shape:")
        print(original_embedding.shape)

        print("\nNoisy Shape:")
        print(noisy_embedding.shape)
        print("\n" + "=" * 60)

        print("Sentence:")
        print(sentence)

        print("\nEntropy:")
        print(round(entropy, 4))

        print("\nTriples:")
        print(triples)

        print("\nKG Text:")
        print(text)

        print("\nSimilarity:")
        print(round(similarity, 4))

        print("\nEncoder Shape:")
        print(
            encoded_data["encoder_outputs"]
            .last_hidden_state
        )

