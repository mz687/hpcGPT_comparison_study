from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

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
    return answer.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1].replace('<|eot_id|>','').replace('<|end_of_text|>','')

def process_ticket_prompts(prompt):
    words = re.split(r'\s+', prompt)

    cache = ''
    res = []
    label = None
    for word in words:
        if word == 'Human:':
            if label:
                if label == 'Human':
                    res.append({"role":"user", 
                                "content":cache})
                else:
                    res.append({"role":"assistant", 
                                "content":cache})
            label = 'Human'
            cache = ''
        elif word == 'Assistant:':
            if label:
                if label == 'Human':
                    res.append({"role":"user", 
                                "content":cache})
                else:
                    res.append({"role":"assistant", 
                                "content":cache})
            label = 'Assistant'
            cache = ''
        else:
            cache = cache + ' ' + word 
    
    ret = []
    prev_tag = None
    idx = 0
    while(idx < len(res)):
        item = res[idx]
        if prev_tag is None or prev_tag != item['role']:
            prev_tag = item['role']
            ret.append(item)
        elif prev_tag == item['role']:
            ret[-1]['content'] += '\n'+item['content']
        else:
            raise Exception('Error in process_prompt')
        idx += 1
        
    return ret
  

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
    parser.add_argument("--RAG_file_path",
                        type=str,
                        required=False,
                        help="Path to the RAG documents")
    parser.add_argument("--ICL_file_path",
                        type=str,
                        required=True,
                        help="Path to the ICL documents")
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
    parser.add_argument("-d","--device",
                        type=str,
                        default='cuda:NVIDIA GH200 120GB',
                        help="Device for the retriever. Choices are 'cpu', 'cuda', 'gpu' ('cpu' requires kompute as backend, which needs compilation from scratch on VISTA)")
    parser.add_argument("--max_context_len",
                        type=int,
                        default=2048,
                        help='Max number of tokens for the input question')
    args = parser.parse_args()
    return args

def load_ICL_files(json_file):

    return [Document(page_content=Path(json_file).read_text())]


def load_files(base_dir):
    '''
    Depth-first traverse and read all the files under 'base_dir'
    '''
    ret = []
    sub_dirs = []
    files = []

    for path in os.listdir(base_dir):
        path = os.path.join(base_dir, path)
        if not os.path.isfile(path):
            sub_dirs.append(path)
        else:
            files.append(path)

    for sub_dir in sub_dirs:
        ret.extend(load_files(sub_dir))
    
    for file in files:
        if ".DS_Store" in file:
            continue
        ret.append(Document(page_content=Path(file).read_text()))

    return ret    

def build_vector_store(docs_list, device):
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add documents to vectorDB
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", 
                                  inference_mode="local",
                                  device=device),
    )

    # Create retriever (return top-3 most relevant doc chunks)
    retriever = vectorstore.as_retriever(k=2)

    return retriever

def get_generator(path,
                  max_context_len,
                  max_new_tokens):
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
            raise Exception(f"model_json ({model_json}) does not exist in path ({path})")
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else int(os.environ['SLURM_PROCID'])
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    init_process(rank, world_size)

    args = get_args()

    # Load in the evaluation dataset
    eval_data_path = args.eval_json_path
    eval_with_answer_path = args.output_file
    with open(eval_data_path, 'r') as f:
        questions = json.load(f)

    # Build ICL vector store
    ICL_file_dir = args.ICL_file_path
    docs_list = load_ICL_files(ICL_file_dir)
    retriever = build_vector_store(docs_list, 
                                   args.device)

    # Load model
    generation_model_id = args.model_path
    print("Using model in {}".format(generation_model_id))

    model, tokenizer = get_generator(generation_model_id,
                                   max_new_tokens=args.max_new_tokens,
                                   max_context_len=args.max_context_len)
    model = model.to("cuda:0")

    # Load chat_template
    chat_template = open('/work/09308/zhengmk/finetune_llama3.1_DL_assignment/DeepSpeedExamples/applications/DeepSpeed-Chat/dschat/utils/data/llama-3.1-instruct-chat-template.jinja', 'r').read()
    chat_template = chat_template.replace('    ', '').replace('\n', '')
    tokenizer.chat_template = chat_template

    # Create Prompts
    system_prompt = "You are a HPC assistant who answers users' tickets."

    icl_prompt = """
    Here are some ticket examples for reference:

    {context} 

    After that is the real conversation and question.

    Please provide a concise answer.
    """

    num_ool_question = 0
    results = []
    for i, ticket in enumerate(questions):
        if i % world_size != rank:
            continue
        
        print(f"[Rank {rank}] i = {i} (total={len(questions)})")    

        res = {}
        prompt = ticket['prompt']

        docs = retriever.invoke(prompt)
        docs_txt = format_docs(docs)

        icl_prompt_template = icl_prompt.format(context=docs_txt)
        
        conversation = process_ticket_prompts(prompt)

        template = [{'role':'system', 'content': system_prompt},
                    {'role':'system', 'content': icl_prompt_template},
                    *conversation],

        icl_prompt_formatted = tokenizer.apply_chat_template(template,
                                                              tokenize=True, 
                                                              return_dict=True, 
                                                              add_generation_prompt=True,
                                                              return_tensors="pt").to(model.device)
        
        num_tokens = icl_prompt_formatted['input_ids'].shape[1] 
        num_ool_question += num_tokens > args.max_context_len

        # Manually truncate the context length to max_context_len
        if num_tokens > args.max_context_len:
            icl_prompt_formatted['input_ids'] = icl_prompt_formatted['input_ids'][:, 0:args.max_context_len-1].reshape(1,-1)
            icl_prompt_formatted['attention_mask'] = icl_prompt_formatted['attention_mask'][:, 0:args.max_context_len-1].reshape(1,-1)
        try:
            # generated = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
            # generated = generated.split('Assistant:')[-1]

            generation = tokenizer.decode(model.generate(**icl_prompt_formatted,
                                            do_sample=False,
                                            repetition_penalty=1.2,
                                            max_new_tokens=args.max_new_tokens)[0])


            res['prompt'] = ticket['prompt']
            res['chosen'] = ticket['chosen']
            res['examples'] = docs_txt
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
    
    # dist.barrier(group=torch.distributed.group.WORLD)

    # results_gathered = [None for _ in range(world_size)]
    # dist.gather_object(results,
    #                    object_gather_list = results_gathered if dist.get_rank() == 0 else None,
    #                    dst=0)

    # results_extended = []
    # for result_gathered in results_gathered:
    #     results_extended.extend(result_gathered)
    # with open(eval_with_answer_path.replace('.json')+f"_rank_{rank}.json", 'w') as f:
    #     json.dump(results_extended, f)

    eval_with_answer_path = eval_with_answer_path.replace('.json','')+f"_rank_{rank}.json"
    with open(eval_with_answer_path, 'w') as f:
        json.dump(results, f)

    print(f"Number of questions with tokens more than 2048: {num_ool_question}")
    print(f"Generated answers are saved to {eval_with_answer_path}")

    # dist.barrier(group=torch.distributed.group.WORLD)
    # dist.destroy_process_group()
