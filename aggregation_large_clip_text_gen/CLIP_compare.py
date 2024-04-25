from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection

# Image embeddings
image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('original_image.png')

inputs = processor(images=image, return_tensors="pt")

outputs = image_model(**inputs)
image_embeds = outputs.image_embeds

# Text embeddings
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# inputs = tokenizer(["two striped cats sleeping and"], padding=True, return_tensors="pt")
inputs = tokenizer(["reference baptist answered robertson gravity testimony fodils halloffame greatly dyslexia karanjohar salads lasers gy receiver âĤ¬ zenith tively skid kens collier shaman poole kennel people mac bwfc herman walker eves ayo oshi reconstruction ible armored nips ^_^ boc schoice toyota ditional debts arers cale walters deer chung kiki pavilion kaa evenings between strive croissant powder and varying tarot illustrator vial"], padding=True, return_tensors="pt")
# inputs = tokenizer(["patriotic elovers transmission Đ kent"], padding=True, return_tensors="pt")
# inputs = tokenizer(["Two cats on a red blanket"], padding=True, return_tensors="pt")
# inputs = tokenizer(["dros doe ney ney haunts benin dino isreal bbs aci gas bed reen rendition abi meda tvmiaw someone ukip dylan sensory ney diamond ney bino pam osborne diamond ignore reen stingray reen ipad mahi mtn corsair aling krystal micah gya diamond nippon lat meister tvmiaw innovation founder alom uab grocery ppm ried atric ri donia 7 mph signals envelope kwa"], padding=True, return_tensors="pt")
# inputs = tokenizer(["pereira africa shirt spoken shut roh ject output python"], padding=True, return_tensors="pt")
# inputs = tokenizer(["dickens rocker doctor dominated watermelon "], padding=True, return_tensors="pt")
# inputs = tokenizer(["stans otic revolutionary , buccaneers"], padding=True, return_tensors="pt")

outputs = text_model(**inputs)
text_embeds = outputs.text_embeds

diff_vec = text_embeds-image_embeds
loss = (diff_vec) @(diff_vec).T
print((diff_vec).shape)
print(f'loss: {loss}')

