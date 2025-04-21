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

### Generate
from langchain_core.messages import HumanMessage, SystemMessage

os.environ['USER_AGENT'] = 'myagent'

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
                        required=True,
                        help="Path to the RAG documents")
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
    pipe = pipeline("text-generation",
                    model=model,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    tokenizer=tokenizer,
                    device="cuda:0",
                    max_new_tokens=max_new_tokens)
    generator = HuggingFacePipeline(pipeline=pipe)

    return generator, tokenizer

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

    # Build RAG vector store
    RAG_file_dir = args.RAG_file_path
    docs_list = load_files(RAG_file_dir)
    retriever = build_vector_store(docs_list, 
                                   args.device)

    # Load in model
    generation_model_id = args.model_path
    print("Using model in {}".format(generation_model_id))

    llm, tokenizer = get_generator(generation_model_id,
                                   max_new_tokens=args.max_new_tokens,
                                   max_context_len=args.max_context_len)

    rag_prompt = """Human: You are an assistant for question-answering tasks. 

    Here is the question:

    {question}

    Here is the context that might be useful for answering the question:

    {context} 

    Think carefully about the above context. 

    Provide an answer to this question using only the above context. 

    Assistant:"""

    # Number of questions with more than 2048 tokens
    num_ool_question = 0
    results = []
    for i, ticket in enumerate(questions):
        if i % world_size != rank:
            continue
        
        print(f"[Rank {rank}] i = {i} (total={len(questions)})")    

        res = {}
        question = ticket['prompt']

        docs = retriever.invoke(question)
        docs_txt = format_docs(docs)

        rag_prompt_formatted = rag_prompt.format(context=docs_txt, 
                                                 question="\nAssistant:".join(question.split('\nAssistant:')[:-1]))

        encoded = tokenizer.encode(rag_prompt_formatted)
        num_tokens = len(encoded)
        
        num_ool_question += num_tokens > 2048

        # Manually truncate the context length to max_context_len
        if num_tokens > args.max_context_len:
            rag_prompt_formatted = tokenizer.decode(encoded[:args.max_context_len-1])
        try:
            generated= llm.invoke([HumanMessage(content=rag_prompt_formatted)])
            generated = generated.split('Assistant:')[-1]

            res['question'] = ticket['prompt']
            res['context'] = docs_txt
            res['augmented'] = rag_prompt_formatted
            res['generated'] = generated
            res['num_tokens_augmented'] = num_tokens

        except:
            raise Exception(f"the question below caused OOM error: {question}\nNumber of tokens: {num_tokens}")
        results.append(res)
    
    eval_with_answer_path = eval_with_answer_path.replace('.json','')+f"_rank_{rank}.json"
    with open(eval_with_answer_path, 'w') as f:
        json.dump(results, f)

    print(f"[rank {rank}] Number of questions with tokens more than 2048: {num_ool_question}")
    print(f"[rank {rank}] Generated answers are saved to {eval_with_answer_path}")

    dist.barrier(group=torch.distributed.group.WORLD)
    dist.destroy_process_group()
