from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
MODEL_NAME = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def decode_text(noisy_embedding):

    encoder_outputs = BaseModelOutput(
        last_hidden_state=noisy_embedding
    )

    generated_ids = model.generate(
        encoder_outputs=encoder_outputs,
        max_length=50
    )

    decoded_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return decoded_text