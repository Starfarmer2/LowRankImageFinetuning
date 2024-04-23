from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection

# Image embeddings
image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('save_im.png')

inputs = processor(images=image, return_tensors="pt")

outputs = image_model(**inputs)
image_embeds = outputs.image_embeds

# Text embeddings
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# inputs = tokenizer(["two striped cats sleeping and cuddling in a red blanket with two remote controls"], padding=True, return_tensors="pt")
# inputs = tokenizer(["Two cats on a red blanket"], padding=True, return_tensors="pt")
inputs = tokenizer(["Two ships at sea"], padding=True, return_tensors="pt")

outputs = text_model(**inputs)
text_embeds = outputs.text_embeds

diff_vec = text_embeds-image_embeds
loss = (diff_vec) @(diff_vec).T
print((diff_vec).shape)
print(f'loss: {loss}')

