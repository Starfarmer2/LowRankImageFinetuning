from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection


import torch
from torch import nn

def find_matching_ids(embedding_vectors, query_vectors, topk=1):
    """
    Find the top-k matching input IDs for each query vector based on cosine similarity.

    Parameters:
    - embedding_vectors (Tensor): The embedding matrix of shape (vocab_size, embedding_dim).
    - query_vectors (Tensor): The vectors to match, of shape (num_queries, embedding_dim).
    - topk (int): Number of top matches to return.

    Returns:
    - indices (Tensor): The matching input IDs for each query vector.
    """
    # Normalize the vectors to unit length
    embedding_vectors = nn.functional.normalize(embedding_vectors, p=2, dim=1)
    query_vectors = nn.functional.normalize(query_vectors, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.matmul(query_vectors, embedding_vectors.t())
    
    # Find the top-k indices
    _, indices = torch.topk(similarity, k=topk, dim=1, largest=True, sorted=True)
    
    return indices.T

# Image embeddings
image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image.save('raw_image.png')

image_inputs = processor(images=image, return_tensors="pt")

outputs = image_model(**image_inputs)
image_embeds = outputs.image_embeds


# Text embeddings
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
for param in text_model.parameters():
    param.requires_grad = False
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

correct_text_inputs = tokenizer(["Fire"], padding=True, return_tensors="pt")
incorrect_text_inputs = tokenizer(["Two dogs on a blue blanket"], padding=True, return_tensors="pt")
print(f'correct_text_inputs: {correct_text_inputs}')

token_embedding_layer = text_model.text_model.embeddings.token_embedding
position_embedding_layer = text_model.text_model.embeddings.position_embedding
encoder_layer = text_model.text_model.encoder
final_norm_layer = text_model.text_model.final_layer_norm
text_projection_layer = text_model.text_projection
print(f'token_embedding_weights: {token_embedding_layer}')
print(f"token_embeddings layer output : {token_embedding_layer(correct_text_inputs['input_ids'])}")

# Find the closest matching input IDs
matching_ids = find_matching_ids(token_embedding_layer.weight.data, token_embedding_layer(correct_text_inputs['input_ids'])[0])
print(f'matching_ids: {matching_ids}')

outputs = text_model(**correct_text_inputs)
correct_token_vectors = token_embedding_layer(correct_text_inputs['input_ids'])
# correct_position_vectors = position_embedding_layer(correct_text_inputs['position_ids'])
print(f'correct_token_vectors: {correct_token_vectors.shape}')
# correct_position_vectors = position_embedding_layer(correct_text_inputs['input_ids'])
input_length = correct_text_inputs['input_ids'].shape[1]
# correct_position_vectors = position_embedding_layer.weight.data[:input_length]
correct_position_vectors = position_embedding_layer(torch.arange(correct_text_inputs['input_ids'].shape[1]).unsqueeze(0))
correct_text_embeds = outputs.text_embeds

outputs = text_model(**incorrect_text_inputs)
incorrect_token_vectors = token_embedding_layer(incorrect_text_inputs['input_ids'])
incorrect_position_vectors = position_embedding_layer(torch.arange(incorrect_text_inputs['input_ids'].shape[1]).unsqueeze(0))
incorrect_text_embeds = outputs.text_embeds

diff_vec = correct_text_embeds-incorrect_text_embeds
print(f'diff_vec: {diff_vec.shape}')

loss = (diff_vec) @(diff_vec).T
print(f'initial loss: {loss}')


# FGSM attack function
def fgsm_attack(original_token_vectors, original_position_vectors, correct_text_embeds, incorrect_text_embeds, encoder_layer, final_norm_layer, text_projection_layer, lr, L=135):
    
    # Iteratively update weights, end condition loss <= L
    new_token_vectors = torch.nn.Parameter(original_token_vectors.clone())
    new_token_vectors.requires_grad = True
    loss = 1000
    # while loss > L:
    for i in range(100):
        new_token_vectors = torch.nn.Parameter(new_token_vectors.clone())
        optimizer = optim.AdamW([new_token_vectors], lr=lr)
        optimizer.zero_grad()
        new_combined_vectors = new_token_vectors + original_position_vectors
        # new_inputs['input_ids'] = new_inputs['input_ids'] * 0 + new_ids.int()

        print(f'new_token_vectors: {new_token_vectors.shape}')
        # print(f'original_position_vectors: {original_position_vectors}')
        outputs = encoder_layer(new_combined_vectors)
        # print(f'outputs_last_hidden_state: {outputs.shape}')
        # print(f'difference of two: {new_combined_vectors - outputs}')
        outputs = final_norm_layer(outputs.last_hidden_state)
        outputs = torch.mean(outputs, dim=1) # pool all the outputs into 512
        outputs = text_projection_layer(outputs)
        new_text_embeds = outputs
        # print(f'new_text_embeds: {new_text_embeds.shape}')
        # print(f'correct_text_embeds: {correct_text_embeds.shape}')
        print(f'print grads:')
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    print(param.grad)
        diff_vec_from_correct = correct_text_embeds-new_text_embeds
        diff_vec_from_incorrect = incorrect_text_embeds-new_text_embeds

        print(f'diff_vec_from_correct: {diff_vec_from_correct.shape}')
        loss = (diff_vec_from_incorrect[0]) @(diff_vec_from_incorrect[0]).T # - (diff_vec_from_correct) @(diff_vec_from_correct).T
        print(f'loss: {loss}')
        loss.backward()

        # Do not modify first and last tokens
        new_token_vectors.grad[0][0] = 0.0
        new_token_vectors.grad[0][-1] = 0.0
        new_token_vectors.grad[0][-2] = 0.0
        # Remove grads for all input vectors
        correct_text_embeds.grad = None
        incorrect_text_embeds.grad = None
        original_token_vectors.grad = None
        original_position_vectors.grad = None

        print(f'new_token_vectors.grad: {new_token_vectors.grad}')
        print(f'original_position.grad: {original_position_vectors.grad}')

        optimizer.step()
        print(f'loss: {loss}')

    # Print text
    text_outputs = find_matching_ids(token_embedding_layer.weight.data, new_token_vectors[0])
    # text_outputs = new_token_vectors.squeeze(0).cpu().detach().numpy()
    print(f'text_outputs: {text_outputs}')
    tokens = tokenizer.convert_ids_to_tokens(text_outputs[0])
    print(f'tokens: {tokens}')



if __name__ == '__main__':
    fgsm_attack(correct_token_vectors, correct_position_vectors, correct_text_embeds, image_embeds, encoder_layer=encoder_layer, final_norm_layer=final_norm_layer, text_projection_layer=text_projection_layer, lr=1)


