from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def triples_to_text(triples):

    text_parts = []

    for s, r, o in triples:
        text_parts.append(f"{s} {r} {o}")

    return " ".join(text_parts)


def encode_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.encoder(
        input_ids=inputs.input_ids
    )
    
    #print(inputs.input_ids.shape)
    #print(outputs.last_hidden_state.shape)

    return outputs.last_hidden_state