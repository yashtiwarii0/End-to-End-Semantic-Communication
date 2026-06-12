from entropy_checker import calculate_entropy
from kg_extractor import extract_triples
from encoder_t5 import triples_to_text, encode_text
from similarity import calculate_similarity
import csv
from channel import transmit_signal
from decoder_t5 import decode_text
from bert_refiner import refine_text

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
        "Decoded_Text",
        "Refined_Text",
        "KG_Similarity",
        "Decoder_Similarity",
        "Refined_Similarity"
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

        kg_similarity = calculate_similarity(
            sentence,
            text
        )
     
        encoded_data = encode_text(text)
        original_embedding = (
        encoded_data["encoder_outputs"]
            .last_hidden_state
        )

        snr_db = 2
        noisy_embedding = transmit_signal(
            original_embedding,
            snr_db
        )

        decoded_text = decode_text(
        noisy_embedding
        )

        decoder_similarity = calculate_similarity(
        sentence,
        decoded_text
        )
        
        refined_output = refine_text(
        decoded_text
        )

        refined_similarity = calculate_similarity(
        sentence,
        refined_output["text"]
        )

        writer.writerow([
            sentence,
            round(entropy, 4),
            str(triples),
            text,
            decoded_text,
            refined_output["text"],
            round(kg_similarity, 4),
            round(decoder_similarity, 4),
            round(refined_similarity, 4)
            ])

        print("\n" + "=" * 60)

        print("Sentence:")
        print(sentence)

        print("\nEntropy:")
        print(round(entropy, 4))

        print("\nTriples:")
        print(triples)

        print("\nKG Text:")
        print(text)

        print("\nSNR:")
        print(f"{snr_db} dB")

        print("\nOriginal Shape:")
        print(original_embedding.shape)

        print("\nNoisy Shape:")
        print(noisy_embedding.shape)
        
        print("\nDecoded Text:")
        print(decoded_text)

        print("\nRefined Output:")
        print(refined_output)

        print("\nKG Similarity:")
        print(round(kg_similarity, 4))

        print("\nDecoder Similarity:")
        print(round(decoder_similarity, 4))
        
        print("\nRefined Similarity:")
        print(round(refined_similarity, 4))

        print("\nEncoder Shape:")
        print(
            encoded_data["encoder_outputs"]
            .last_hidden_state.shape
        )

