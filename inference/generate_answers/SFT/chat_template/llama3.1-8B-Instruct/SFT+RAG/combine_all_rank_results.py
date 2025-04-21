import json
import os

path = '/work/09308/zhengmk/finetune_llama3.1_DL_assignment/DeepSpeedExamples/applications/DeepSpeed-Chat/inference/SC25/eval/chat_template/llama3.1-8B-Instruct/SFT+RAG'
rank_results = [os.path.join(path, file) for file in os.listdir(path) if '_rank_' in file and '.json' in file]

results = []
for rank_result in rank_results:
    with open(rank_result, 'r') as f:
        print(rank_result)
        rank_result_json = json.load(f)
    results.extend(rank_result_json)

output_path = os.path.join(path, 'eval_answered_k_2_distributed.json')
with open(output_path, 'w') as f:
    json.dump(results,f)

print("Combined results saved to ", output_path)