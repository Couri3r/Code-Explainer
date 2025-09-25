import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

model_path = r"checkpoint-13752"


tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded successfully on device: {device}")


def explain_code(code_snippet: str):

    prefix = "explain code: "
    
    inputs = tokenizer.encode(
        prefix + code_snippet,
        return_tensors="pt",
        truncation=True
    ).to(device)
    
    outputs = model.generate(
        inputs,
        max_length=100,
        num_beams=5,
        early_stopping=True
    )
    
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation




# my_code = my_code = """
# def is_prime(num):
#     if num < 2:
#         return False
#     for i in range(2, int(num**0.5) + 1):
#         if num % i == 0:
#             return False
#     return True
# """
# generated_explanation = explain_code(my_code)


# print(generated_explanation)
