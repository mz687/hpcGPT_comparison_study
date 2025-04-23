import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

with open("eval_results_chatbot.json", "r") as f:
    qa_items = json.load(f)

with open("/scratch/09979/esther_lh/hpcgpt/zmk/data/eval_categorized.json", "r") as f1:
    category_items = json.load(f1)

try:
    with open("/work/09308/zhengmk/finetune_llama3.1_DL_assignment/DeepSpeedExamples/applications/DeepSpeed-Chat/inference/SC25/random_sample_for_manual/random_samples_examples_single_round_keys.json", 'r', encoding='utf-8') as f2:
        clean_prompts = json.load(f2)
    
    print(f"Successfully loaded {len(clean_prompts)} clean prompts from JSON file")
    if len(clean_prompts) > 0:
        print(f"Example prompt: {clean_prompts[0]}")
except Exception as e:
    print(f"Error loading clean prompts file: {e}")
    clean_prompts = []

qa_dict = {item["prompt"]: item for item in qa_items}

matched_items = []
for item in category_items:
    prompt = item["prompt"]
    if prompt in clean_prompts and prompt in qa_dict:
        matched_items.append({
            "query": prompt,
            "response": qa_dict[prompt]["generated"],
            "chosen": qa_dict[prompt]["chosen"],
            "categorized": item.get("categorized","unknown")
        })
print(len(matched_items), "matched!")

query = [item["query"] for item in matched_items]
actual_output = [item["response"] for item in matched_items]
expected_output = [item["chosen"] for item in matched_items]

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

print("Embedding queries...")
query_embed = model.encode(query, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
print("Embedding actual outputs...")
actual_output_embed = model.encode(actual_output, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
print("Embedding expected output...")
expected_output_embed = model.encode(expected_output, batch_size=32, convert_to_tensor=True, show_progress_bar=True)

qa_similarity = util.cos_sim(query_embed, actual_output_embed)
outputs_similarity = util.cos_sim(expected_output_embed, actual_output_embed)

category_scores_qa = defaultdict(float)
category_scores_outputs = defaultdict(float)
category_counts = defaultdict(int)

qa_scores = []
outputs_scores = []

for i in tqdm(range(len(matched_items))):
    category = matched_items[i]["categorized"]
    
    qa_score = qa_similarity[i][i].item()
    outputs_score = outputs_similarity[i][i].item()

    qa_scores.append(qa_score)
    outputs_scores.append(outputs_score)

    category_scores_qa[category] += qa_score
    category_scores_outputs[category] += outputs_score
    category_counts[category] += 1

print("Calculate average Cosine Similarity of each categories.")
for category in sorted(category_counts.keys()):
    count = category_counts[category]
    qa_avg = category_scores_qa[category] / count
    outputs_avg = category_scores_outputs[category] / count
    print(f"{category}: QA sim = {qa_avg:.4f}, Output sim = {outputs_avg:.4f}")
