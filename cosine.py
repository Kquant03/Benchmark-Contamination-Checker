import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a cross-encoder model fine-tuned for semantic similarity
model_name = "cross-encoder/stsb-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def compute_similarity(text_pairs):
    inputs = tokenizer(text_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.squeeze().cpu().numpy()
    return scores

def detect_cheating(benchmark_dataset, evaluated_dataset, similarity_threshold=1.20, batch_size=1024):
    cheating_examples = []
    total = len(benchmark_dataset) * len(evaluated_dataset)
    
    with tqdm(total=total, desc="Detecting cheating") as pbar:
        for i in range(0, len(evaluated_dataset), batch_size):
            eval_batch = evaluated_dataset[i:i+batch_size]
            eval_texts = [ex["conversations"][0]["value"] + " " + ex["conversations"][1]["value"] for ex in eval_batch]
            
            for benchmark_ex in benchmark_dataset:
                benchmark_text = benchmark_ex["question"] + " " + benchmark_ex["answer"]
                text_pairs = [(benchmark_text, eval_text) for eval_text in eval_texts]
                similarities = compute_similarity(text_pairs)
                
                for j, similarity in enumerate(similarities):
                    if similarity > similarity_threshold:
                        cheating_examples.append({
                            "evaluated_example": eval_batch[j],
                            "benchmark_example": benchmark_ex,
                            "similarity_score": similarity
                        })
                
                pbar.update(batch_size)
    
    cheating_percentage = (len(cheating_examples) / len(evaluated_dataset)) * 100
    return cheating_percentage, cheating_examples

# Load datasets
print("Loading benchmark dataset...")
with open("/home/kquant/Desktop/detect-pretrain-code-contamination-master/src/mmlu_final.jsonl", "r") as file:
    benchmark_dataset = [json.loads(line) for line in file]

print("Loading evaluated dataset...")
with open("/home/kquant/Documents/Code/nemotron-datagen/generated_conversations.jsonl", "r") as file:
    evaluated_dataset = [json.loads(line) for line in file]

print(f"Number of benchmark examples: {len(benchmark_dataset)}")
print(f"Number of evaluated examples: {len(evaluated_dataset)}")

if not benchmark_dataset or not evaluated_dataset:
    print("Error: One or both datasets are empty. Please check your input files.")
else:
    # Detect cheating
    similarity_threshold = 1.2  # Adjust this threshold as needed
    batch_size = 1024  # Adjust the batch size as per your system's memory and performance
    cheating_percentage, cheating_examples = detect_cheating(benchmark_dataset, evaluated_dataset, similarity_threshold, batch_size)
    
    print(f"Cheating Percentage: {cheating_percentage:.2f}%")
    
    # Save cheating examples
    print("Saving cheating examples...")
    with open("cheating_examples.jsonl", "w") as file:
        for example in cheating_examples:
            json.dump(example, file)
            file.write("\n")
    
    print(f"Cheating examples saved to 'cheating_examples.jsonl'.")