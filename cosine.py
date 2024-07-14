import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Set the start method to 'spawn' inside the __main__ block

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

def process_batch(batch_data):
    eval_batch, benchmark_dataset, similarity_threshold = batch_data
    cheating_examples = []
    
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
    
    return cheating_examples

def detect_cheating(benchmark_dataset, evaluated_dataset, similarity_threshold=0.8, batch_size=8, num_processes=4):
    total = len(benchmark_dataset) * len(evaluated_dataset)
    
    pool = mp.Pool(processes=num_processes)
    
    batches = [(evaluated_dataset[i:i+batch_size], benchmark_dataset, similarity_threshold)
               for i in range(0, len(evaluated_dataset), batch_size)]
    
    results = []
    with tqdm(total=total, desc="Detecting cheating") as pbar:
        for batch_result in pool.imap_unordered(process_batch, batches):
            results.extend(batch_result)
            pbar.update(batch_size * len(benchmark_dataset))
    
    pool.close()
    pool.join()
    
    cheating_percentage = (len(results) / len(evaluated_dataset)) * 100
    return cheating_percentage, results

if __name__ == '__main__':
    # Load datasets
    print("Loading benchmark dataset...")
    with open("/home/kquant/Desktop/detect-pretrain-code-contamination-master/src/arc_c_test_final.jsonl", "r") as file:
        benchmark_dataset = [json.loads(line) for line in file]

    # Randomly select 100 samples from the benchmark dataset
    random.shuffle(benchmark_dataset)
    benchmark_dataset = benchmark_dataset[:100]

    print("Loading evaluated dataset...")
    with open("/home/kquant/Desktop/detect-pretrain-code-contamination-master/src/generated_conversations.jsonl", "r") as file:
        evaluated_dataset = [json.loads(line) for line in file]

    print(f"Number of benchmark examples: {len(benchmark_dataset)}")
    print(f"Number of evaluated examples: {len(evaluated_dataset)}")

    if not benchmark_dataset or not evaluated_dataset:
        print("Error: One or both datasets are empty. Please check your input files.")
    else:
        # Detect cheating
        similarity_threshold = 1.2  # Adjust this threshold as needed
        batch_size = 8  # Adjust the batch size as per your system's memory and performance
        num_processes = 4  # Adjust the number of processes based on your system's capabilities
        cheating_percentage, cheating_examples = detect_cheating(benchmark_dataset, evaluated_dataset, similarity_threshold, batch_size, num_processes)
        
        print(f"Cheating Percentage: {cheating_percentage:.2f}%")
        
        # Save cheating examples
        print("Saving cheating examples...")
        with open("cheating_examples.jsonl", "w") as file:
            for example in cheating_examples:
                json.dump(example, file)
                file.write("\n")
        
        print(f"Cheating examples saved to 'cheating_examples.jsonl'.")
