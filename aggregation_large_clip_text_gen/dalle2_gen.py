from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-2",
  prompt="a white siamese cat",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url