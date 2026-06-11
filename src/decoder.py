from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def decode_text(encoded_data):

    generated_ids = model.generate(
        encoded_data["input_ids"],
        max_length=50
    )

    decoded_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return decoded_text