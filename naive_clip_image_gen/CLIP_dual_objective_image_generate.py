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
image.save('raw_image.png')

image_inputs = processor(images=image, return_tensors="pt")

outputs = image_model(**image_inputs)
image_embeds = outputs.image_embeds

# Text embeddings (Dual objectives, original and new captions)
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

correct_text_inputs = tokenizer(["Two cats on a red blanket"], padding=True, return_tensors="pt")
incorrect_text_inputs = tokenizer(["Two ships at sea"], padding=True, return_tensors="pt")

outputs = text_model(**correct_text_inputs)
correct_text_embeds = outputs.text_embeds

outputs = text_model(**incorrect_text_inputs)
incorrect_text_embeds = outputs.text_embeds

diff_vec_from_correct = correct_text_embeds-image_embeds
diff_vec_from_incorrect = incorrect_text_embeds-image_embeds

loss = (diff_vec_from_incorrect) @(diff_vec_from_incorrect).T - (diff_vec_from_correct) @(diff_vec_from_correct).T
print(f'initial loss: {loss}')
print(f'image_inputs: {image_inputs.pixel_values.shape}')


# FGSM attack function
def fgsm_attack(image, correct_text_embeds, incorrect_text_embeds, model, lr, L=-20):
    image = torch.nn.Parameter(image.clone())
    # image = torch.nn.Parameter(torch.rand(image.size()))
    image.requires_grad = True
    optimizer = optim.AdamW([image], lr=lr)

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
        outputs = image_model(image)
        image_embeds = outputs.image_embeds
        diff_vec_from_correct = correct_text_embeds-image_embeds
        diff_vec_from_incorrect = incorrect_text_embeds-image_embeds

        loss = (diff_vec_from_incorrect) @(diff_vec_from_incorrect).T - (diff_vec_from_correct) @(diff_vec_from_correct).T
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f'loss: {loss}')

    # Save image
    save_im = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    save_im_max = save_im.max()
    save_im_min = save_im.min()
    save_im = (save_im - save_im_min) / (save_im_max-save_im_min)
    save_im = Image.fromarray((save_im * 255).astype('uint8'))
    save_im.save("save_im.png")

fgsm_attack(image_inputs.pixel_values, correct_text_embeds, incorrect_text_embeds, image_model, 0.1)

