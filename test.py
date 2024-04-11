import torch
import torch.nn.functional as F

# Example output from a model (log probabilities) and true labels
log_probs = torch.log(torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]))
labels = torch.tensor([2, 1])

loss = F.nll_loss(log_probs, labels, reduction='none')
print(log_probs)
print(loss)