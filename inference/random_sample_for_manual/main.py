import json
import os
import argparse
import random
import pandas as pd
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deduplication_file_path",
                        type=str,
                        help='The key in this file will be used to deduplicate.')
    parser.add_argument("--generated_answers",
                        action='append',
                        required=True,
                        help="Paths to the json files of the generated answers")
    parser.add_argument("--num_samples",
                        type=int,
                        default=200,
                        help='Number of samples')
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help='Absolute path to the output json file')
    args = parser.parse_args()
    return args

def random_sample(keys,
                  json_files_dict,
                  num_samples):
    random.seed(12345)
    
    # random_key_idx = [random.randint(0, len(keys)-1) for _ in range(num_samples)]
    random_key_idx = list(random.sample(range(len(keys)), num_samples))
    random_keys = [keys[idx] for idx in random_key_idx]

    val_to_key = {}
    ret = {}
    chosen = {}

    for idx, random_key in enumerate(random_keys):
        temp = []
        val_to_key_temp = {}
        for file_path, json_file in json_files_dict.items():
            prompts = [ticket['prompt'] if 'prompt' in ticket else ticket['question'] for ticket in json_file]
            assert random_key in prompts
            for ticket in json_file:
                question = ticket['prompt'] if 'prompt' in ticket else ticket['question']
                if question == random_key:
                    if 'chosen' in ticket:
                        chosen[question] = ticket['chosen']
                    temp.append(ticket['generated'])
                    val_to_key_temp[file_path] = ticket['generated']

        ret[random_key] = temp
        val_to_key[random_key] = val_to_key_temp
        assert len(temp) == len(json_files_dict)

    return ret, val_to_key, chosen

if __name__ == '__main__':
    args = get_args()

    # Read in all the json files
    generated_answers = {}
    keys_sets = {}
    for file in args.generated_answers:
        with open(file, 'r') as f:
            generated_answer = json.load(f)
            generated_answers[file] = generated_answer
        
        keys_sets[file] = set()
        # use 'prompt' in this file as the keys
        for ticket in generated_answer:
            question = ticket['prompt'] if 'prompt' in ticket else ticket['question']
            if len(re.findall('Human:|Assistant:', question)) == 2 and re.search('Human:', question) and re.search('Assistant:', question):
                keys_sets[file].add(question)

    keys = keys_sets[args.generated_answers[0]]
    for key, keys_set in keys_sets.items():
        keys = keys.intersection(keys_set)
    keys = list(keys)

    single_round_keys_file_name = args.output_file.replace('.json', '') + '_keys.json'
    with open(single_round_keys_file_name, 'w') as f:
        json.dump(keys, f)

    with open(args.deduplication_file_path, 'r') as f:
        deduplicate_file = json.load(f)
    for key in deduplicate_file.keys():
        if key in keys:
            keys.remove(key)

    print("Total number of target QA pairs: ", len(keys))


    ret, val_to_key, chosen = random_sample(keys,
                                            generated_answers, 
                                            args.num_samples)
    
    with open(args.output_file, 'w') as f:
        json.dump(ret, f)
    print("Random sampled examples are saved to {}".format(args.output_file))

    answer_columns = [f"Answer {i}" for i in range(len(args.generated_answers))]
    df_dict = {}
    df_dict['Question'] = []
    df_dict['Reference_answer'] = []
    for key, vals in ret.items():
        df_dict['Question'].append(key)
        df_dict['Reference_answer'].append(chosen[key])
        for i, val in enumerate(vals):
            if answer_columns[i] not in df_dict:
                df_dict[answer_columns[i]] = []
            df_dict[answer_columns[i]].append(val)

    df = pd.DataFrame(df_dict)
    df.to_csv(args.output_file.replace('.json','.csv'))

    val_to_key_path = args.output_file.replace('.json', '')+'_val_to_key.json'
    with open(val_to_key_path, 'w') as f:
        json.dump(val_to_key, f)
    print("The value to keys are saved to {}".format(val_to_key_path))