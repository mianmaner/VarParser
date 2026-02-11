from varparser.cache import VarCache
from varparser.prompt import llm_querying
from varparser.utils import correct_message
from varparser.eval import evaluator

from tqdm import tqdm
import pandas as pd
import argparse
import logging
import time
import yaml
import os

# nltk.download('words')
# nltk.download('wordnet')

# logging.basicConfig(filename='main.log', level=logging.INFO,
#                     format='%(message)s')


class VarParser:
    def __init__(self, dataset, model, data_type, shot):
        self.dataset = dataset
        self.data_type = data_type
        self.model = model
        self.shot = shot

    def parse(self):
        logging.info(f"start parsing {self.dataset}...")
        varcache = VarCache(dataset=self.dataset, shot=self.shot)
        sampled_data = self.read_sampled_data(self.dataset)
        varcache.extract_vars(sampled_data)

        log_data = self.read_data()
        log_messages = log_data['Content'].tolist()
        ground_truth = log_data['EventTemplate'].tolist()

        invoc_num = 0
        invoc_time = 0
        token_consumed = 0

        index_result = []
        start_time = time.time()

        for log_message in tqdm(log_messages, total=len(log_messages)):
            index, success = varcache.match(log_message)
            if not success:
                var_prompt = varcache.adaptive_var_selection(log_message)
                reference = varcache.get_reference(log_message)
                print(f"query: {log_message}")
                t0 = time.time()
                response, tokens = llm_querying(
                    query_log=log_message, var_prompt=var_prompt, reference=reference)
                invoc_time += time.time() - t0
                invoc_num += 1
                token_consumed += tokens
                print(f"answer: {response}")
                index = varcache.insert(log_message, response)

            index_result.append(index)

        end_time = time.time()
        parse_time = end_time - start_time
        # print(f"Parse time: {parse_time:.4f}s")
        # print(invoc_num, token_consumed/invoc_num, invoc_time)

        parsed_result = map(lambda x: varcache.templates[x], index_result)
        # print(varcache.templates)
        # print(set(index_result))
        parsed_csv = pd.DataFrame(
            {"parsed_result": parsed_result, "ground_truth": ground_truth})
        parsed_csv.to_csv(
            f"parsed/{self.dataset}_{self.shot}_parsed.csv", index=False)

        avg_token = token_consumed / invoc_num if invoc_num != 0 else 0
        evaluator(self.dataset, self.model, self.shot, parse_time,
                  invoc_num, avg_token, invoc_time)
        return

    def read_data(self):
        data_dir = f"dataset/{self.data_type}_dataset"
        data_path = os.path.join(
            data_dir, self.dataset, f"{self.dataset}_{self.data_type}.log_structured.csv")
        log_df = pd.read_csv(data_path)
        log_data = log_df[['Content', 'EventTemplate']]
        log_data = log_data.applymap(correct_message)
        return log_data

    def read_sampled_data(self, dataset):
        data_dir = "sample"
        data_path = os.path.join(data_dir, f"{dataset}_sampled_examples.csv")
        sampled_df = pd.read_csv(data_path)
        return sampled_df


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--data_type", type=str, default="full")
    parser.add_argument("--shot", type=int, default=3)
    args = parser.parse_args()

    if args.dataset != "all":
        datasets = [args.dataset]
    else:
        datasets = [
            "Apache",
            "BGL",
            "Hadoop",
            "HDFS",
            "HealthApp",
            "HPC",
            "Linux",
            "Mac",
            "OpenSSH",
            "OpenStack",
            "Proxifier",
            "Spark",
            "Thunderbird",
            "Zookeeper"
        ]

    for dataset in datasets:
        logparser = VarParser(
            dataset=dataset,
            model=config['model'],
            data_type=args.data_type,
            shot=args.shot)
        logparser.parse()
