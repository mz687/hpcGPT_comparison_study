# from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
import argparse

### Generate
from langchain_core.messages import HumanMessage, SystemMessage

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
    generator = pipeline("text-generation",
                         model=model,
                         model_kwargs={"torch_dtype": torch.bfloat16},
                         tokenizer=tokenizer,
                         device="cuda:0",
                         max_new_tokens=max_new_tokens)
    return generator, tokenizer

if __name__ == '__main__':
    args = get_args()

    # generation_model_id= '/scratch/09308/zhengmk/finetune_llama3.1_8b_Dahoas_hull_hh_rlhf_local_context_2048_hpcgpt_v3/epoch_3/converted/'
    generation_model_id = args.model_path
    print("Using model in {}".format(generation_model_id))

    llm, tokenizer = get_generator(generation_model_id,
                                   max_context_len=args.max_context_len,
                                   max_new_tokens=args.max_new_tokens)

    eval_data_path = args.eval_json_path
    eval_with_answer_path = args.output_file

    with open(eval_data_path, 'r') as f:
        questions = json.load(f)

    # Number of questions with more than 2048 tokens
    num_ool_question = 0
    for ticket in tqdm(questions):
        question = ticket['prompt']
        encoded = tokenizer.encode(question)
        num_tokens = len(encoded)
        
        num_ool_question += num_tokens > 2048

        # Manually truncate the context length to max_context_len
        # tokenizer's truncation still causes CUDA OOM error for extremely long questions (e.g. 28000)
        if num_tokens > args.max_context_len:
            question = tokenizer.decode(encoded[:args.max_context_len-1])

        try:
            ticket['generated'] = llm(question)[0]['generated_text'].replace(question, '')
            ticket['num_tokens_og'] = num_tokens
        except:
            raise Exception(f"the question below caused OOM error: {question}\nNumber of tokens: {num_tokens}")

    with open(eval_with_answer_path, 'w') as f:
        json.dump(questions, f)
    
    print(f"Number of questions with tokens more than 2048: {num_ool_question}")
    print(f"Generated answers are saved to {eval_with_answer_path}")
