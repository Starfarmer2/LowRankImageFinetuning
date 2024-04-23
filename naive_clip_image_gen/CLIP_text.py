from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")

input_ids = inputs.input_ids
input_ids[0][2] = 1300


outputs = model(input_ids=input_ids, attention_mask = inputs.attention_mask)
text_embeds = outputs.text_embeds

print(text_embeds)
print(inputs['input_ids'])