# from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
import argparse

import torch
import torch.distributed as dist

import re

### Generate
from langchain_core.messages import HumanMessage, SystemMessage

import combine 

def init_process(rank, world_size):
    dist.init_process_group("nccl", 
    rank=rank, 
    world_size=world_size)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Directory containing trained actor model")
    parser.add_argument("--json_path",
                        type=str,
                        default="/work/09308/zhengmk/finetune_llama3.1_DL_assignment/DeepSpeedExamples/applications/DeepSpeed-Chat/data/eval.json")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help='Absolute path to the output json file')
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=1024,
                        help="Maximum new tokens to generate per response",)
    parser.add_argument("-t","--temperature",
                        type=float,
                        default=0.,
                        help="Generation temperature")
    parser.add_argument("--max_context_len",
                        type=int,
                        default=2048,
                        help='Max number of tokens for the input question')
    parser.add_argument("--postprocess_only",
                        action='store_true',
                        help='If set, will not run inference and only perform post-processing')
    args = parser.parse_args()
    return args

def get_model_response(generator, 
                       user_input, 
                       max_new_tokens,
                       temperature):
    if temperature == 0.0:
        response = generator(user_input, 
                             max_new_tokens=max_new_tokens)
    else:
        response = generator(user_input,
                             max_new_tokens = max_new_tokens,
                             temperature = temperature,
                             do_sample = True)
    return response

def get_generator(path,
                  max_new_tokens):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, 
                                                  fast_tokenizer=True)

    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.truncation_side = "right" 

    model_config = AutoConfig.from_pretrained(path)
    model_class = AutoModelForCausalLM.from_config(model_config)
    model = model_class.from_pretrained(path,
                                        from_tf=bool(".ckpt" in path),
                                        config=model_config).half()

    model.resize_token_embeddings(len(tokenizer))
    generator = pipeline("text-generation",
                         model=model,
                         model_kwargs={"torch_dtype": torch.bfloat16},
                         tokenizer=tokenizer,
                         device="cuda:0",
                         max_new_tokens=max_new_tokens)
    return generator, tokenizer

def inference(args):
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else int(os.environ['SLURM_PROCID'])
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    generation_model_id = args.model_path
    if rank == 0:
        print("Using model in {}".format(generation_model_id))

    llm, tokenizer = get_generator(generation_model_id,
                                   max_new_tokens=args.max_new_tokens)

    data_path = args.json_path
    data_with_answer_path = args.output_file

    with open(data_path, 'r') as f:
        questions = json.load(f)

    prompt = """You are an assistant for classifying HPC tickets into four categories: policy, technical, real-time machine status, and debugging. Here are the brief descriptions for each of them:
    - Policy-related questions refer to those that can be solve by reading the documentation of HPC manual, including how to register for an account, how to login to the HPC machines, how to reset password, how to replace email address, request for software license, project allocation and etc.
    - Technical questions refer to those that need help from HPC techical staff. Example questions include "how to install a python package", "which branch/version of the software should be used for deployment", and "how to use a specific software on an HPC machine".
    - Machine status question refer to those which need information about the real-time machine status, including questions related to date for scheduled maintanence, complaining about system down, complaining about slow access speed to file system. 
    - Debugging question should have code error message attached to the question.

    Here are the question and the answer:

    {question}

    Think carefully. Give the category of the question above only. No reasons needed.
    <think>\n
    """
    
    results = []

    for i, ticket in enumerate(questions):
        if i % world_size != rank:
            continue
        
        print("[rank {}] {}/{}".format(rank, i, len(questions)), flush=True)

        result = {}

        question = ticket['prompt']
        answer = ticket['chosen']
        qa = question + '\n\n' + answer
        prompt_formatted = prompt.format(question=qa)

        encoded = tokenizer.encode(prompt_formatted)
        num_tokens = len(encoded)

        # skip too long prompt
        if num_tokens > 10_000:
            result['category'] = 'OOM'
            continue

        result['prompt+chosen'] = qa
        
        llm_res = get_model_response(llm,
                                    prompt_formatted,
                                    args.max_new_tokens,
                                    args.temperature)[0]['generated_text']
        result['category'] = llm_res.replace(prompt_formatted, '')

        results.append(result)
    
    data_with_answer_path = data_with_answer_path.replace(".json","")+f"_rank_{rank}.json"
    with open(data_with_answer_path, 'w') as f:
        json.dump(results, f)
    print('Final result has been saved to ', data_with_answer_path)

def post_process_category(file_path):
    json_file = None

    with open(file_path, 'r') as f:
        json_file = json.load(f)
    assert json_file is not None

    num_policy = 0
    num_technical = 0
    num_machine_status = 0
    num_debugging = 0

    for i, item in enumerate(json_file):
        if re.search(r'[P|p]olicy', item['category']):
            item['categorized'] = 'Policy'
            num_policy += 1
        elif re.search('[D|d]ebug', item['category']):
            item['categorized'] = 'Debugging'
            num_debugging += 1
        elif re.search('[T|t]echnical', item['category']):
            item['categorized'] = 'Technical'
            num_technical += 1
        elif re.search('[M|m]achine [S|s]tatus', item['category']):
            item['categorized'] = 'Machine Status'
            num_machine_status += 1
        else:
            item['categorized'] = ''
        
        if len(item['categorized']) == 0:
            json_file.pop(i)
    
    output_file_path = file_path.replace('.json','')+'_postprocessed.json'
    with open(output_file_path, 'w') as f:
        json.dump(json_file, f)

    print(f"Number of policy: {num_policy} / {len(json_file)} ({num_policy/len(json_file)*100:.2f}%)")
    print(f"Number of debugging: {num_debugging} / {len(json_file)} ({num_debugging/len(json_file)*100:.2f}%)")
    print(f"Number of technical: {num_technical} / {len(json_file)} ({num_technical/len(json_file)*100:.2f}%)")
    print(f"Number of machine status: {num_machine_status} / {len(json_file)} ({num_machine_status/len(json_file)*100:.2f}%)")
    print(f"Processed json file has been saved to {output_file_path}")

if __name__ == '__main__':
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else int(os.environ['SLURM_PROCID'])
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    init_process(rank, world_size)

    args = get_args()
    
    if not args.postprocess_only:
        inference(args)

    if rank == 0:
        combine.main(args.output_file.split('_')[0])
        post_process_category(args.output_file)