from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection

# Image embeddings
image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_inputs = processor(images=image, return_tensors="pt")

outputs = image_model(**image_inputs)
image_embeds = outputs.image_embeds

# Text embeddings
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

text_inputs = tokenizer(["A green dog"], padding=True, return_tensors="pt")

outputs = text_model(**text_inputs)
text_embeds = outputs.text_embeds

diff_vec = text_embeds-image_embeds
loss = (diff_vec) @(diff_vec).T
print((diff_vec).shape)
print(f'initial loss: {loss}')
print(f'image_inputs: {image_inputs.pixel_values.shape}')


# FGSM attack function
def fgsm_attack(text, image_embeds, model, lr, L=40):
    text = torch.nn.Parameter(text.clone().to(dtype=torch.float32))
    text.requires_grad = True
    optimizer = optim.AdamW([text], lr=lr)

    # # Iteratively update weights
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     output = model(image)
    #     loss = criterion(output, label)
    #     loss.backward()
    #     optimizer.step()

    # Iteratively update weights, end condition loss <= L
    loss = 1000
    while loss > L:
        optimizer.zero_grad()
        outputs = text_model(input_ids=text.round().int(), attention_mask = text_inputs.attention_mask)
        text_embeds = outputs.text_embeds
        diff_vec = text_embeds-image_embeds
        loss = (diff_vec) @(diff_vec).T
        loss.backward(retain_graph=True)
        print(text.grad)
        optimizer.step()
        print(f'loss: {loss}')

    # Save image
    print(f'learned text: {tokenizer.decode(text[0])}')

fgsm_attack(text_inputs.input_ids, image_embeds, text_model, 10)

