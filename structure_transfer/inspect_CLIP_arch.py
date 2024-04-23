from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print(model.vision_model)