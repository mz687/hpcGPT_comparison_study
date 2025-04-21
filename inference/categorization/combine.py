import json
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        required=True,
                        help="mode can be either 'train' or 'eval'")
    args = parser.parse_args()
    return args

def main(mode):
    path = './'
    rank_results = [os.path.join(path, file) for file in os.listdir(path) if '_rank_' in file and '.json' in file and mode in file ]

    results = []
    for rank_result in rank_results:
        with open(rank_result, 'r') as f:
            print(rank_result)
            rank_result_json = json.load(f)
        results.extend(rank_result_json)
        os.remove(rank_result)

    output_path = os.path.join(path, f'{mode}_categoried.json')
    with open(output_path, 'w') as f:
        json.dump(results,f)

    print("Combined results saved to ", output_path)

if __name__ == '__main__':
    args = get_args()
    main(args.mode)