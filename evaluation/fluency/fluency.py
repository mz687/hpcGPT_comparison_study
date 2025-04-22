import json
import os
from tqdm import tqdm
import argparse

from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from fluency_metric import FluencyMetric  
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
        return self.model_name

    def generate(self, prompt: str, schema=None, max_length=None) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Move inputs to the model's device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Use max_new_tokens for better control
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
        # Extract and process the JSON
        try:
            # Extract JSON from output
            json_str = self._extract_json(output)
            json_data = json.loads(json_str) if json_str else {}
            # If schema is provided and it's a class (not a dict type)
            if schema and hasattr(schema, '__annotations__'):
                # Create empty object with required schema keys
                if not json_data:
                    json_data = {}
                # Ensure all required keys exist
                for key in schema.__annotations__:
                    if key not in json_data:
                        if key in ['statements', 'verdicts']:
                            json_data[key] = []
                        elif key == 'reason':
                            json_data[key] = "No reason provided"
                        else:
                            json_data[key] = ""
                # Return an instance of the schema class
                return schema(**json_data)
            # Return the dictionary if no schema or it's not a class
            return json_data
        except Exception as e:
            print(f"Error in JSON processing: {e}")
            # Create and return proper schema instance if needed
            if schema and hasattr(schema, '__annotations__'):
                empty_data = {
                    key: ([] if key in ['statements', 'verdicts'] else "Error in processing")
                    for key in schema.__annotations__
                }
                return schema(**empty_data)
            # Fallback to empty dict with statements
            return {"statements": []}

    def _extract_json(self, text):
        """Extract JSON object from text"""
        # Try to find JSON object with brackets
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            try:
                json_str = text[json_start:json_end]
                # Validate it's valid JSON
                json.loads(json_str)
                return json_str
            except:
                pass
        # If simple extraction fails, try more advanced methods
        import re
        pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in reversed(matches):  # Start from the last match
                try:
                    json.loads(match)
                    return match
                except:
                    continue
        # Return empty JSON if all methods fail
        return '{}'

def load_dataset(file_path: str):
    """Load dataset from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Extract query and response pairs
        test_cases = []
        for item in data:
            if isinstance(item, dict) and "question" in item and "generated" in item:
                test_cases.append({
                    "query": item["question"],
                    "response": item["generated"]
                })
        print(f"Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def evaluate_fluency(test_cases, eval_model, threshold=0.7):
    # Initialize the fluency metric
    fluency_metric = FluencyMetric(
        threshold=threshold,
        model=eval_model,
        async_mode=True  
    )

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    results = []
    
    # Process each test case
    for i, case in enumerate(tqdm(test_cases, desc="Evaluating test cases")):
        if i % world_size != rank:
            continue

        query = case["query"]
        response = case["response"]

        test_case = LLMTestCase(
            input=query,
            actual_output=response
        )

        # Evaluate the test case
        try:
            evaluation_result = evaluate([test_case], [fluency_metric])
            test_result = evaluation_result.test_results[0]
            metric_data = test_result.metrics_data[0]

            score = metric_data.score
            reason = metric_data.reason

        except Exception as e:
            print(f"Error during evaluation of case #{i}: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Store result
        results.append({
            "query": query,
            "response": response,
            "fluency_score": score,
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
    parser = argparse.ArgumentParser(description="Evaluate fluency of responses")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="Model for evaluation")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON file to process")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory for results")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Fluency threshold")
    
    args = parser.parse_args()
   
    # init torch.distributed
    rank = init(os.environ['RANK']) if 'RANK' in os.environ else int(os.environ['SLURM_PROCID'])
    world_size = int(os.getenv('WORLD_SIZE', 1))
    init_distributed_process(rank, world_size)

    # output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # evaluation model
    if dist.get_rank() == 0:
        print(f"Loading evaluation model: {args.model}")
    eval_llm = LocalLLM(args.model)
    
    if not os.path.exists(args.input_file):
        if dist.get_rank() == 0:
            print(f"Input file not found: {args.input_file}")
        return
    
    file_name = os.path.basename(args.input_file)
    if dist.get_rank() == 0:    
        print(f"\nProcessing {file_name}...")
    
    # Load and evaluate data
    test_cases = load_dataset(args.input_file)
    if not test_cases:
        if dist.get_rank() == 0:
            print(f"No valid test cases in {file_name}")
        return
    
    results = evaluate_fluency(test_cases, eval_llm, args.threshold)
        
    # Calculate average score
    avg_score = sum(r["fluency_score"] for r in results) / len(results) if results else 0
    print(f"Average fluency score for {file_name}: {avg_score:.4f}")
    
    # Save results
    output_path = os.path.join(args.output_dir, f"fluency_{file_name}")
    with open(output_path.replace('.json', '') + f'_rank_{rank}.json', "w", encoding="utf-8") as f:
        json.dump({
            "file_name": file_name,
            "average_score": avg_score,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")

    cleanup()

if __name__ == "__main__":
    main()
