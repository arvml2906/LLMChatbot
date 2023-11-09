# %%
"""
# Import Libraries
"""

# %%
import re
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

# %%
import torch
import torch.nn as nn
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from model import GPTLanguageModel  # Import  model class

# %%
"""
# Hyperparameters
"""

# %%
# hyperparameters

max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
batch_size = 16  # independent sequences processed in parallel
block_size = 32  # maximum context length for predictions
# ------------

torch.manual_seed(1337)

# %%


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# %%
"""
# Loading the Dataset
"""

# %%

with open('binary_operation_fine_shuffled_file.csv', 'r', encoding='utf-8') as f:
    text = f.read()

# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


# %%
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# %%
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%

# %%
"""
# Train the model
"""

# %%
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%


def generate_prompt_response(prompt, model, max_new_tokens):
    model.eval()
    input_tokens = encode(prompt)
    context = torch.tensor(
        input_tokens,
        dtype=torch.long,
        device=device).unsqueeze(0)
    generated_tokens = model.generate(
        context, max_new_tokens=max_new_tokens)[0].tolist()
    generated_response = decode(generated_tokens)

    # Use regular expression to match the answer format (e.g., "3+4=7"or
    # "3x4=12")
    match = re.search(r'\d+\s*([-+\/xX])\s*\d+\s*=\s*\d+', generated_response)
    if match:
        answer = match.group(0)
        return answer

    return "Answer not found"


trained_model = m


# %%
prompt = "hey, what's 3+13?"  # The expected result is 16
max_new_tokens = 15  # Maximum number of tokens in the generated response

# Generate response based on the prompt
generated_response = generate_prompt_response(
    prompt, trained_model, max_new_tokens)
print(f"Input Prompt: {prompt}\nGenerated Response: {generated_response}")

# %%
prompt = "What is 12x10?"  # The expected result is 120
max_new_tokens = 15  # Maximum number of tokens in the generated response

# Generate response based on the prompt
generated_response = generate_prompt_response(
    prompt, trained_model, max_new_tokens)
print(f"Input Prompt: {prompt}\nGenerated Response: {generated_response}")

# %%
prompt = "What is 12/6?"  # The expected result is 2
max_new_tokens = 15  # Maximum number of tokens in the generated response

# Generate response based on the prompt
generated_response = generate_prompt_response(
    prompt, trained_model, max_new_tokens)
print(f"Input Prompt: {prompt}\nGenerated Response: {generated_response}")

# %%
prompt = "value of 15+5?"  # The expected result is 20
max_new_tokens = 15  # Maximum number of tokens in the generated response

# Generate response based on the prompt
generated_response = generate_prompt_response(
    prompt, trained_model, max_new_tokens)
print(f"Input Prompt: {prompt}\nGenerated Response: {generated_response}")
