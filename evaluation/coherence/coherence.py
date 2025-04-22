import json
import os
import re
from tqdm import tqdm
import argparse

from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from coherence_metric import CoherenceMetric
from deepeval.models import DeepEvalBaseLLM

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import torch.distributed as dist

def init_distributed_process(rank, world_size):
    torch.distributed.init_process_group(backend='nccl',
                                         rank = rank,
                                         world_size = world_size)
def cleanup():
    torch.distributed.destroy_process_group()

class LocalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the evaluation model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=device
            )
            print(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_model_name(self):
        """Return the model name"""
        return self.model_name

    def generate(self, prompt: str, schema=None, max_length=None) -> str:
        """Generate text using max_new_tokens instead of max_length"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during generation: {e}")
            return "{}"

    async def a_generate(self, prompt: str, schema=None, max_length=None):
        """Asynchronous inference"""
        output = self.generate(prompt)
        try:
            json_str = self._extract_json(output)
            json_data = json.loads(json_str) if json_str else {}
            
            if schema and hasattr(schema, '__annotations__'):
                if not json_data:
                    json_data = {}
                for key in schema.__annotations__:
                    if key not in json_data:
                        if key in ['statements', 'verdicts']:
                            json_data[key] = []
                        elif key == 'reason':
                            json_data[key] = "No reason provided"
                        else:
                            json_data[key] = ""
                return schema(**json_data)
            return json_data
        except Exception as e:
            print(f"Error in JSON processing: {e}")
            if schema and hasattr(schema, '__annotations__'):
                empty_data = {
                    key: ([] if key in ['statements', 'verdicts'] else "Error in processing")
                    for key in schema.__annotations__
                }
                return schema(**empty_data)
            return {"statements": []}

    def _extract_json(self, text):
        """Extract JSON object from text"""
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            try:
                json_str = text[json_start:json_end]
                json.loads(json_str)
                return json_str
            except:
                pass
        
        pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in reversed(matches):
                try:
                    json.loads(match)
                    return match
                except:
                    continue
        return '{}'

def load_datasets(generated_file: str, ground_truth_file: str):
    """Load and match datasets from two files"""
    try:
        # Load generated outputs
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
        
        # Load ground truth
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        
        # Create dictionaries for faster matching
        prompt_to_generated = {}
        for item in generated_data:
            if isinstance(item, dict) and "question" in item and "generated" in item:
                prompt_to_generated[item["question"]] = item["generated"]
        
        # Create test cases by matching prompts
        test_cases = []
        matched_count = 0
        
        for gt_item in ground_truth_data:
            if isinstance(gt_item, dict) and "prompt" in gt_item and "chosen" in gt_item:
                prompt = gt_item["prompt"]
                chosen = gt_item["chosen"]
                
                if prompt in prompt_to_generated:
                    matched_count += 1
                    test_cases.append({
                        "query": prompt,
                        "response": prompt_to_generated[prompt],
                        "expected_output": chosen
                    })
        
        print(f"Loaded {len(test_cases)} matched test cases out of {len(ground_truth_data)} ground truth items")
        return test_cases
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return []

def evaluate_coherence(test_cases, eval_model, threshold=0.7):
    """Evaluate coherence between actual output and expected output"""
    coherence_metric = CoherenceMetric(
        threshold=threshold,
        model=eval_model,
        async_mode=True 
    )

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    results = []
    
    for i, case in enumerate(tqdm(test_cases, desc="Evaluating test cases")):
        if i % world_size != rank:
            continue

        test_case = LLMTestCase(
            input=case["query"],
            actual_output=case["response"],
            expected_output=case["expected_output"]
        )

        try:
            evaluation_result = evaluate([test_case], [coherence_metric])
            
            test_result = evaluation_result.test_results[0]
            metric_data = test_result.metrics_data[0]
            
            score = metric_data.score
            reason = metric_data.reason
        
        except Exception as e:
            print(f"Error during evaluation of case #{i}: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        results.append({
            "query": case["query"],
            "response": case["response"],
            "expected_output": case["expected_output"],
            "coherence_score": score,
            "reason": reason
        })

    #dist.barrier()
    #results_gathered = [None for _ in range(world_size)]
    #dist.gather_object(results, results_gathered if dist.get_rank() == 0 else None, dst=0)

    #res = []
    #if dist.get_rank() == 0:
    #    for result in results_gathered:
    #        res.extend(result)
    #dist.barrier()

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate coherence between generated and ground truth outputs")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="Model for evaluation")
    parser.add_argument("--generated_file", type=str, required=True,
                        help="JSON file with generated outputs")
    parser.add_argument("--ground_truth_file", type=str, required=True,
                        help="JSON file with ground truth (chosen) outputs")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory for results")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Coherence threshold")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # init torch.distributed
    rank = init(os.environ['RANK']) if 'RANK' in os.environ else int(os.environ['SLURM_PROCID'])
    world_size = int(os.getenv('WORLD_SIZE', 1))
    init_distributed_process(rank, world_size)

    os.makedirs(args.output_dir, exist_ok=True)
    
    if dist.get_rank() == 0:
        print(f"Loading evaluation model: {args.model}")
    eval_llm = LocalLLM(args.model)
    
    if not os.path.exists(args.generated_file):
        if dist.get_rank() == 0:
            print(f"Generated file not found: {args.generated_file}")
        return
    
    if not os.path.exists(args.ground_truth_file):
        if dist.get_rank() == 0:
            print(f"Ground truth file not found: {args.ground_truth_file}")
        return

    if dist.get_rank() == 0: 
        print(f"\nMatching data between {os.path.basename(args.generated_file)} and {os.path.basename(args.ground_truth_file)}...")
    
    test_cases = load_datasets(args.generated_file, args.ground_truth_file)
    if not test_cases:
        if dist.get_rank() == 0:
            print("No matched test cases found")
        return
    
    results = evaluate_coherence(test_cases, eval_llm, args.threshold)
    
    if not results:
        if dist.get_rank() == 0:
            print("No evaluation results generated")
        return
   
    avg_score = sum(r["coherence_score"] for r in results) / len(results) if results else 0
    print(f"Average coherence score: {avg_score:.4f}")
    
    output_filename = f"coherence_{os.path.basename(args.generated_file)}"
    output_path = os.path.join(args.output_dir, output_filename)
    with open(output_path.replace('.json', '') + f'_rank_{rank}.json', "w", encoding="utf-8") as f:
        json.dump({
            "generated_file": args.generated_file,
            "ground_truth_file": args.ground_truth_file,
            "average_score": avg_score,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")

    cleanup()

if __name__ == "__main__":
    main()
