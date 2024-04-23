from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection

# Image embeddings
# image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_model = CLIPVisionModelWithProjection.from_pretrained("mlunar/models/")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image.save('raw_image.png')

# Forward pass on style objective

# List to store the outputs of the layers you are interested in
intermediate_outputs = []

def hook_function(module, input, output):
    # Append the output of this layer to the list
    intermediate_outputs.append(output)

hook1 = image_model.vision_model.embeddings.patch_embedding.register_forward_hook(hook_function)
hook2 = image_model.vision_model.encoder.layers[0].register_forward_hook(hook_function)
hook3 = image_model.vision_model.encoder.layers[2].register_forward_hook(hook_function)

image_inputs = processor(images=image, return_tensors="pt")

outputs = image_model(**image_inputs)
image_embeds = outputs.image_embeds
# Remove hooks after use
hook1.remove()
hook2.remove()

print(f'image_inputs: {image_inputs.pixel_values.shape}')
print(f'intermediate_outputs: {intermediate_outputs}')




# # FGSM attack function
# def fgsm_attack(image, text_embeds, model, lr, L=-0.7):
#     image = torch.nn.Parameter(image.clone())
#     # image = torch.nn.Parameter(torch.rand(image.size()))
#     image.requires_grad = True
#     optimizer = optim.AdamW([image], lr=lr)

#     # # Iteratively update weights
#     # for epoch in range(num_epochs):
#     #     optimizer.zero_grad()
#     #     output = model(image)
#     #     loss = criterion(output, label)
#     #     loss.backward()
#     #     optimizer.step()

#     # Iteratively update weights, end condition loss <= L
#     loss = 1000
#     while loss > L:
#         optimizer.zero_grad()
#         outputs = image_model(image)
#         image_embeds = outputs.image_embeds

#         # Cosine similary negated
#         loss = -(text_embeds @ image_embeds.T) / text_embeds.norm(p=2) / image_embeds.norm(p=2)
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         print(f'loss: {loss}')

#     # Save image
#     save_im = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
#     save_im_max = save_im.max()
#     save_im_min = save_im.min()
#     save_im = (save_im - save_im_min) / (save_im_max-save_im_min)
#     save_im = Image.fromarray((save_im * 255).astype('uint8'))
#     save_im.save("save_im.png")

# fgsm_attack(image_inputs.pixel_values, text_embeds, image_model, 0.1)

