# from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore

from pathlib import Path

import torch.distributed as dist
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import random
import json
import os
from tqdm import tqdm
import argparse
import re

### Generate
from langchain_core.messages import HumanMessage, SystemMessage

os.environ['USER_AGENT'] = 'myagent'

def post_process(answer):
    return answer.split('<|start_header_id|>assistant<|end_header_id|>')[-1].replace('<|eot_id|>','')

def process_ticket_prompts(prompt):
    words = re.split(r'\s+', prompt)

    cache = ''
    res = []
    label = None
    for word in words:
        if word == 'Human:':
            if label:
                if label == 'Human':
                    # res.append(HumanMessage(content=cache))
                    res.append({"role":"user",
                                "content":cache})
                else:
                    # res.append(AIMessage(content=cache))
                    res.append({"role":"assistant",
                                "content":cache})
            label = 'Human'
            cache = ''
        elif word == 'Assistant:':
            if label:
                if label == 'Human':
                    # res.append(HumanMessage(content=cache))
                    res.append({"role":"user",
                                "content":cache})
                else:
                    # res.append(AIMessage(content=cache))
                    res.append({"role":"assistant",
                                "content":cache})
            label = 'Assistant'
            cache = ''
        else:
            cache = cache + ' ' + word
    return res

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Directory containing trained actor model")
    parser.add_argument("--eval_json_path",
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
    args = parser.parse_args()
    return args

def get_generator(path,
                  max_new_tokens,
                  max_context_len):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      max_length=max_new_tokens,
                                                      fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, 
                                                  max_length=max_context_len, 
                                                  fast_tokenizer=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right" 

    model_config = AutoConfig.from_pretrained(path)
    model_class = AutoModelForCausalLM.from_config(model_config)
    model = model_class.from_pretrained(path,
                                        from_tf=bool(".ckpt" in path),
                                        config=model_config).half()

    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

if __name__ == '__main__':
    args = get_args()

    # generation_model_id= '/scratch/09308/zhengmk/finetune_llama3.1_8b_Dahoas_hull_hh_rlhf_local_context_2048_hpcgpt_v3/epoch_3/converted/'
    generation_model_id = args.model_path
    print("Using model in {}".format(generation_model_id))

    model, tokenizer = get_generator(generation_model_id,
                                   max_context_len=args.max_context_len,
                                   max_new_tokens=args.max_new_tokens)
    model = model.to("cuda:0")

    eval_data_path = args.eval_json_path
    eval_with_answer_path = args.output_file

    with open(eval_data_path, 'r') as f:
        questions = json.load(f)

    # Create Prompts
    system_prompt = "You are a HPC assistant who answers users' tickets. Here is a conversation and question for you:"

    system_prompt_end = "Please think carefully and provide a concise answer."

    # Number of questions with more than 2048 tokens
    num_ool_question = 0
    results =[]
    for i, ticket in enumerate(questions):
        # if i % world_size != rank:
        #     continue

        # print(f"[Rank {rank}] i = {i} (total={len(questions)})")

        res = {}
        prompt = ticket['prompt']

        conversation = process_ticket_prompts(prompt)

        template = [{'role':'system', 'content': system_prompt},
                    *conversation,
                    {'role':'system', 'content': system_prompt_end}]
                   

        prompt_formatted = tokenizer.apply_chat_template(template,
                                                              tokenize=True,
                                                              return_dict=True,
                                                              add_generation_prompt=True,
                                                              return_tensors="pt").to(model.device)

        num_tokens = prompt_formatted['input_ids'].shape[1]
        num_ool_question += num_tokens > args.max_context_len

        # Manually truncate the context length to max_context_len
        if num_tokens > args.max_context_len:
            prompt_formatted = prompt_formatted[:args.max_context_len-1]
        try:
            # generated = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
            # generated = generated.split('Assistant:')[-1]

            generation = tokenizer.decode(model.generate(**prompt_formatted,
                                            do_sample=False,
                                            repetition_penalty=1.2,
                                            max_new_tokens=args.max_new_tokens)[0])

            #print(">>>> Answer: \n", generation)
            #print('-'*100)

            res['prompt'] = ticket['prompt']
            res['chosen'] = ticket['chosen']
            res['generated'] = post_process(generation)
            res['num_tokens_augmented'] = num_tokens

        except Exception as e:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>")
            print(f"Failed at index {i}")
            print(f">>>Prompt: {prompt}")
            try:
                print(f">>>Generation: {generation}")
            except:
                print("!!!No generation produced.")
            print(f"Error: {e}")
            continue

        results.append(res)

    with open(eval_with_answer_path, 'w') as f:
        json.dump(results, f)

    print(f"Number of questions with tokens more than 2048: {num_ool_question}")
    print(f"Generated answers are saved to {eval_with_answer_path}")
