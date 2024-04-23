from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

text = "Hello, world!"
text = "thenorth nunes charger recession deny baw"
text = "swithhump preference muscles dragged blanket"
text = ".. malescursciencefiction husky blanket"
tokens = tokenizer(text, truncation=True, padding=True)
print(tokens)

tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'])
print(f'reconstructed text: {tokens}')