from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection


import torch
from torch import nn

# Filter tokenizer vocab to only include tokens ending with </w>
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
vocab = tokenizer.get_vocab()
tokens_with_w = {token: idx for token, idx in vocab.items() if token.endswith('</w>') or token.endswith('<|startoftext|>') or token.endswith('<|endoftext|>')}
# tokens_with_w = {token: idx for token, idx in vocab.items()}

def reconstruct_text(tokens):
    # Initialize an empty list to hold words
    words = []
    current_word = ''
    
    # Iterate over each token in the list
    for token in tokens:
        if token.endswith('<|startoftext|>') or token.endswith('<|endoftext|>'):
            continue
        if token.endswith('</w>'):
            # If the token ends with '</w>', it is the end of a word
            current_word += token[:-4]  # Remove '</w>' and add to current word
            words.append(current_word)  # Add the complete word to the list
            current_word = ''  # Reset current word
        else:
            # If not ending in '</w>', continue forming the current word
            current_word += token
    
    # Join all words with spaces to form the reconstructed text
    return ' '.join(words)

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

    # ID[indices]
    IDs = list(tokens_with_w.values())
    

    # Only include token strings ending with </w> for token embedding table
    weight_table = embedding_vectors[IDs]

    # Normalize the vectors to unit length
    embedding_vectors = nn.functional.normalize(weight_table, p=2, dim=1)
    query_vectors = query_vectors[0]
    query_vectors = nn.functional.normalize(query_vectors, p=2, dim=1)

    print(f'weight_table: {weight_table.shape}')
    print(f'query_vectors: {query_vectors.shape}')
    
    # Compute cosine similarity
    similarity = torch.matmul(query_vectors, embedding_vectors.t())
    
    print(f'similarity: {similarity.shape}')
    # Find the top-k indices
    _, indices = torch.topk(similarity, k=1, dim=1, largest=True, sorted=True)
    
    # Convert indices to IDs
    # print (f'indices.T: {indices.T}')
    # print(f'[idx] for idx in indices.T[0]]: {[[idx] for idx in indices.T]}')
    # 
    matching_ids = [IDs[idx] for idx in indices]
    print(f'matching_ids: {matching_ids}')
    return matching_ids

# Image embeddings
image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('tee.png')
# image.save('raw_image.png')

image_inputs = processor(images=image, return_tensors="pt")

outputs = image_model(**image_inputs)
image_embeds = outputs.image_embeds


# Text embeddings
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
for param in text_model.parameters():
    param.requires_grad = False

# correct_text_inputs = tokenizer(["Fire Eating James Ben Water Jazz"], padding=True, return_tensors="pt")
# correct_text_inputs = tokenizer(["Five cars red"], padding=True, return_tensors="pt")
correct_text_inputs = tokenizer(["Calculator dog eat cat yum"], padding=True, return_tensors="pt")
incorrect_text_inputs = tokenizer(["Two dogs on a blue blanket"], padding=True, return_tensors="pt")
# print(f'correct_text_inputs: {correct_text_inputs}')

token_embedding_layer = text_model.text_model.embeddings.token_embedding
# print(f'EMBEDDING_VECTORS: {token_embedding_layer.weight.data.shape}')
position_embedding_layer = text_model.text_model.embeddings.position_embedding
encoder_layer = text_model.text_model.encoder
final_norm_layer = text_model.text_model.final_layer_norm
text_projection_layer = text_model.text_projection
# print(f'token_embedding_weights: {token_embedding_layer}')
print(f"token_embeddings layer output : {token_embedding_layer(correct_text_inputs['input_ids'])[0]}")


# Find the closest matching input IDs
matching_ids = find_matching_ids(token_embedding_layer.weight.data, token_embedding_layer(correct_text_inputs['input_ids']))
# print(f'matching_ids: {matching_ids}')

outputs = text_model(**correct_text_inputs)
correct_token_vectors = token_embedding_layer(correct_text_inputs['input_ids'])
# correct_position_vectors = position_embedding_layer(correct_text_inputs['position_ids'])
# print(f'correct_token_vectors: {correct_token_vectors.shape}')
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
# print(f'diff_vec: {diff_vec.shape}')

loss = (diff_vec) @(diff_vec).T
print(f'initial loss: {loss}')


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - vec1 (Tensor): The first vector.
    - vec2 (Tensor): The second vector.

    Returns:
    - similarity (Tensor): The cosine similarity between the two vectors.
    """
    # Normalize the vectors to unit length
    vec1 = nn.functional.normalize(vec1, p=2, dim=1)
    vec2 = nn.functional.normalize(vec2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.matmul(vec1, vec2.t())
    
    return similarity


# FGSM attack function
def fgsm_attack(original_token_vectors, original_position_vectors, correct_text_embeds, incorrect_text_embeds, encoder_layer, final_norm_layer, text_projection_layer, lr, L=2.98):#L=-0.68):
    new_token_vectors = torch.nn.Parameter(original_token_vectors.clone())
    new_token_vectors.requires_grad = True
    optimizer = optim.Adam([new_token_vectors], lr=lr)
    
    # Iteratively update weights, end condition loss <= L
    loss = 1000
    # Example: Using StepLR
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    while loss > L:
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
        print(f'new_text_embeds: {new_text_embeds.shape}')
        # print(f'new_text_embeds: {new_text_embeds.shape}')
        # print(f'correct_text_embeds: {correct_text_embeds.shape}')
        # print(f'print grads:')
        # for param_group in optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param.grad is not None:
        #             print(param.grad)
        #             print(param.grad[param.grad != 0])
                    
        diff_vec_from_correct = correct_text_embeds-new_text_embeds
        
        diff_vec_from_incorrect = incorrect_text_embeds-new_text_embeds

        # print(f'diff_vec_from_correct: {diff_vec_from_correct.shape}')
        # loss = -cosine_similarity(incorrect_text_embeds, new_text_embeds) + (diff_vec_from_incorrect @ diff_vec_from_incorrect.T)/100
        loss = (diff_vec_from_incorrect @ diff_vec_from_incorrect.T)/100
        # print(f'loss: {loss}')
        loss.backward(retain_graph=True)

        # Do not modify first and last tokens
        new_token_vectors.grad[0][0] = 0.0
        new_token_vectors.grad[0][-1] = 0.0
        # new_token_vectors.grad[0][-2] = 0.0
        # new_token_vectors.grad[0][-2] = 0.0
        # Remove grads for all input vectors
        correct_text_embeds.grad = None
        incorrect_text_embeds.grad = None
        original_token_vectors.grad = None
        original_position_vectors.grad = None

        # print(f'new_token_vectors.grad: {new_token_vectors.grad}')
        # print(f'original_position.grad: {original_position_vectors.grad}')

        optimizer.step()
        scheduler.step()
        print("Loss {}; Learning Rate: {}".format(loss, scheduler.get_last_lr()))

    # Print text
    text_outputs = find_matching_ids(token_embedding_layer.weight.data, new_token_vectors)
    # text_outputs = new_token_vectors.squeeze(0).cpu().detach().numpy()
    print(f'text_outputs: {text_outputs}')
    tokens = tokenizer.convert_ids_to_tokens(text_outputs)
    print(f'tokens: {tokens}')
    print(f'text: {reconstruct_text(tokens)}')



if __name__ == '__main__':
    fgsm_attack(correct_token_vectors, correct_position_vectors, correct_text_embeds, image_embeds, encoder_layer=encoder_layer, final_norm_layer=final_norm_layer, text_projection_layer=text_projection_layer, lr=100)
    None


# 0 9 8 6

# [34234.0 234234,0 234324,0 ]
# [34234.0 234234,0 234324,0 ]
# [34234.0 234234,0 234324,0 ]
# [34234.0 234234,0 234324,0 ]