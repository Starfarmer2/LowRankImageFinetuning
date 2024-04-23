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
structure_objective_image = Image.open(requests.get(url, stream=True).raw)
structure_objective_image.save('structure_objective.png')

# Forward pass on structure objective

# List to store the outputs of the layers you are interested in
global structure_intermediate_outputs 
structure_intermediate_outputs = []

def structure_hook_function(module, input, output):
    # Append the output of this layer to the list
    structure_intermediate_outputs.append(output)

# structure_hook1 = image_model.vision_model.embeddings.patch_embedding.register_forward_hook(structure_hook_function)
structure_hook1 = image_model.vision_model.encoder.layers[4].register_forward_hook(structure_hook_function)
structure_hook2 = image_model.vision_model.encoder.layers[0].register_forward_hook(structure_hook_function)
structure_hook3 = image_model.vision_model.encoder.layers[2].register_forward_hook(structure_hook_function)

structure_image_inputs = processor(images=structure_objective_image, return_tensors="pt")

structure_outputs = image_model(**structure_image_inputs)
structure_image_embeds = structure_outputs.image_embeds

# Remove hooks after use
structure_hook1.remove()
structure_hook2.remove()
structure_hook3.remove()


# Original image
# url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/HST-SM4.jpeg/1200px-HST-SM4.jpg"
url = "https://source.unsplash.com/random/1024x768?landscape"
original_image = Image.open(requests.get(url, stream=True).raw)
original_image.save('original_image.png')

original_image_inputs = processor(images=original_image, return_tensors="pt")

global original_intermediate_outputs 
original_intermediate_outputs = []
def original_hook_function(module, input, output):
    # Append the output of this layer to the list
    original_intermediate_outputs.append(output)


# Structure transfer function
def structure_transfer(image, structure_intermediate_outputs, lr, L=-2.8):
    global original_intermediate_outputs 
    image = torch.nn.Parameter(image.clone())
    # image = torch.nn.Parameter(torch.rand(image.size()))
    image.requires_grad = True
    optimizer = optim.AdamW([image], lr=lr)

    # original_hook1 = image_model.vision_model.embeddings.patch_embedding.register_forward_hook(original_hook_function)
    original_hook1 = image_model.vision_model.encoder.layers[4].register_forward_hook(original_hook_function)
    original_hook2 = image_model.vision_model.encoder.layers[0].register_forward_hook(original_hook_function)
    original_hook3 = image_model.vision_model.encoder.layers[2].register_forward_hook(original_hook_function)

    # structure_outputs = image_model(**structure_image_inputs)
    # structure_image_embeds = structure_outputs.image_embeds

    # # Iteratively update weights
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     output = model(image)
    #     loss = criterion(output, label)
    #     loss.backward()
    #     optimizer.step()

    # Iteratively update weights, end condition loss <= L
    loss = 0
    while loss > L:
        optimizer.zero_grad()
        outputs = image_model(image)

        # Cosine similary negated
        loss = torch.tensor(0,requires_grad=True, dtype=torch.float32)
        # print(f'original_intermediate: {original_intermediate_outputs}')
        # print(f'structure_intermediate_outputs: {structure_intermediate_outputs}')
        for structure_features, original_features in zip(structure_intermediate_outputs, original_intermediate_outputs):
            structure_flat = structure_features[0].flatten()
            original_flat = original_features[0].flatten()
            # print(structure_flat)
            # print(original_flat)
            loss = loss -(structure_flat @ original_flat.T) / structure_flat.norm(p=2) / original_flat.norm(p=2) 
        loss.backward(retain_graph=True)
        # print(image.grad)
        optimizer.step()
        print(f'loss: {loss}')
        original_intermediate_outputs = []

    # Save image
    save_im = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    save_im_max = save_im.max()
    save_im_min = save_im.min()
    save_im = (save_im - save_im_min) / (save_im_max-save_im_min)
    save_im = Image.fromarray((save_im * 255).astype('uint8'))
    save_im.save("structure_transfer.png")

    # Remove hooks after use
    original_hook1.remove()
    original_hook2.remove()
    original_hook3.remove()

structure_transfer(original_image_inputs.pixel_values, structure_intermediate_outputs, 0.1)

