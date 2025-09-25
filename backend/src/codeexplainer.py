from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataset import ds 
from datasets import load_dataset 
from transformers import DataCollatorForSeq2Seq






model_name = "Salesforce/codet5p-220m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
prefix = "explain code:"

def preprocess_function(examples):
    inputs = [prefix + c for c in examples["code"]]
    model_inputs = tokenizer(inputs, max_length = 1000, truncation=True)

    labels = tokenizer(text_target=examples["comment"], max_length = 500, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = ds.map(preprocess_function, batched=True)


training_args = Seq2SeqTrainingArguments(
    output_dir="./code_explainer_model",
    learning_rate=2e-5,
    per_device_train_batch_size=2,       
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True, 
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model, 
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  
)

trainer.train()

code_snippet = """def get_file_extension(filename):
    # a simple function to get the file extension
    return filename.split('.')[-1]
"""

inputs = tokenizer.encode(prefix + code_snippet, return_tensors="pt", truncation=True)

outputs = model.generate(
    inputs,
    max_length=60,
    min_length=10,
    num_beams=5,
    early_stopping=True,
)
explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Code: \n", code_snippet)
print("\nGenerated Explanation (from fine-tuned model): \n", explanation)