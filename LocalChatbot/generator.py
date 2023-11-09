import torch
from gpt_model import GPTLanguageModel  # Import your GPTLanguageModel class
import re

with open('binary_operation_fine_shuffled_file.csv', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


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


# Load the trained model
model_path = "trained_model.pth"  # Update with the actual path to your saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = GPTLanguageModel().to(device)
trained_model.load_state_dict(torch.load(model_path, map_location=device))
trained_model.eval()

# Define the prompt
while True:

    prompt = input("Enter a prompt:")  # The expected result is 12
    max_new_tokens = 15  # Maximum number of tokens in the generated response

    # Generate response based on the prompt
    generated_response = generate_prompt_response(
        prompt, trained_model, max_new_tokens)
    print(f"Input Prompt: {prompt}\nGenerated Response: {generated_response}")
