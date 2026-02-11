from varparser.utils import tokenize, jaccard_similarity, mask_message, correct_message

from collections import defaultdict
from nltk.corpus import words
from tqdm import tqdm
import pandas as pd
import calendar
import time
import nltk
import re
import os

# nltk.download('words')

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


class Sampler:
    def __init__(self, log_df, max_shot):
        self.common_words = words.words()
        self.log_df = log_df
        self.max_shot = max_shot

    def sample(self):
        st_time = time.time()

        # Deduplicate logs by uncommon tokens
        grouped_df = self.deduplicating(self.log_df)

        # Cluster uncommon tokens
        uncommon_tokens_list = grouped_df['UncommonTokens'].tolist()
        all_uncommon_tokens = [
            token for tokens in uncommon_tokens_list for token in tokens]
        all_uncommon_tokens = list(set(all_uncommon_tokens))
        token_clusters, mapping = self.token_cluster(all_uncommon_tokens)

        # Calculate contribution and sample
        log_tokens_list = tokenize(grouped_df['Content'])
        token_positions, log_scores = self.calculate_scores(
            log_tokens_list, uncommon_tokens_list, mapping)
        sampled_df = self.greedy_diversified_sampling(
            grouped_df, token_positions, uncommon_tokens_list, log_scores, mapping)

        ed_time = time.time()
        sample_time = ed_time - st_time
        print(f"Sampling time: {sample_time:.4f}s")

        return sampled_df, sample_time

    def get_uncommon_tokens(self, log_tokens_list):
        uncommon_tokens_list = []
        for log_tokens in log_tokens_list:
            uncommon_tokens = []
            for token in log_tokens:
                if token not in self.common_words:
                    uncommon_tokens.append(token)
            uncommon_tokens_list.append(list(set(uncommon_tokens)))
        return uncommon_tokens_list

    def deduplicating(self, log_df):
        # Mask content and group
        log_df['MaskContent'] = log_df['Content'].map(mask_message)
        unique_df = log_df.drop_duplicates(subset='MaskContent')

        # Filter content and group
        temp_df = log_df.copy()
        temp_df['Format'] = temp_df['Content'].map(
            lambda x: re.sub(r'[a-zA-Z0-9]', '', x))
        temp_df = temp_df.drop_duplicates(subset='Format')

        # Tokenize and filter out common words
        all_tokens = tokenize(temp_df['Content'])
        uncommon_tokens = {content: [token for token in tokens if token not in self.common_words]
                           for content, tokens in zip(temp_df['Content'], all_tokens)}

        temp_df['UncommonTokens'] = temp_df['Content'].map(uncommon_tokens)
        grouped_df = temp_df.drop_duplicates(subset='UncommonTokens')
        return grouped_df

    def calculate_scores(self, log_tokens_list, uncommon_tokens_list, mapping):
        token_positions = {}

        for index, uncommon_tokens in enumerate(uncommon_tokens_list):
            for token in uncommon_tokens:
                token_pos = log_tokens_list[index].index(token)
                pos_index = mapping[token]
                if pos_index not in token_positions:
                    token_positions[pos_index] = []
                if token_pos not in token_positions[pos_index]:
                    token_positions[pos_index].append(token_pos)

        log_scores = []
        for uncommon_tokens in uncommon_tokens_list:
            score = 0
            for token in uncommon_tokens:
                index = mapping[token]
                score += len(token_positions[index])
            log_scores.append(score)

        # print(token_positions)
        return token_positions, log_scores

    def token_cluster(self, tokens):
        # Cluster tokens based on Jaccard distance
        token_clusters = []
        token_to_cluster = {}

        for token in tokens:
            if not token_clusters:
                token_clusters.append([token])
                token_to_cluster[token] = 0
            else:
                for idx, cluster in enumerate(token_clusters):
                    if token in cluster:
                        break
                    if jaccard_similarity(token, cluster[0]) >= 0.8:
                        cluster.append(token)
                        token_to_cluster[token] = idx
                        break
                else:
                    # no matched cluster
                    token_clusters.append([token])
                    token_to_cluster[token] = len(token_clusters) - 1
        return token_clusters, token_to_cluster

    def greedy_diversified_sampling(self, grouped_df, token_positions, uncommon_tokens_list, log_scores, mapping):
        sampled_cluster_index = []
        sampled_indices = []

        sampled_num = min(self.max_shot, len(uncommon_tokens_list))
        while len(sampled_indices) < sampled_num:
            # sorted the logs by score
            indexed_scores = list(enumerate(log_scores))
            sorted_indices = sorted(
                indexed_scores, key=lambda x: x[1], reverse=True)

            for index, _ in sorted_indices:
                if index not in sampled_indices:
                    sampled_indices.append(index)
                    # update token score
                    for token in uncommon_tokens_list[index]:
                        cluster_index = mapping[token]
                        sampled_cluster_index.append(cluster_index)
                    # sampled_tokens.extend(uncommon_tokens_list[index])
                    break

            # update log score
            for i in range(len(log_scores)):
                if i in sampled_indices:
                    log_scores[i] = 0
                    continue
                score = 0
                for token in uncommon_tokens_list[i]:
                    cluster_index = mapping[token]
                    if cluster_index in sampled_cluster_index:
                        continue
                    score += len(token_positions[cluster_index])
                log_scores[i] = score

        sampled_df = grouped_df.iloc[sampled_indices]
        return sampled_df


if __name__ == "__main__":

    data_dir = "dataset/full_dataset"
    whole_time = 0
    max_shot = 32
    for dataset in datasets:
        print(f"Sampling {dataset} dataset")

        data_path = os.path.join(
            data_dir, dataset, f"{dataset}_full.log_structured.csv")
        log_df = pd.read_csv(data_path)

        k_rate = 0.2
        length = k_rate * len(log_df)
        log_df = log_df.head(int(length))

        sampler = Sampler(log_df, max_shot)
        sampled_df, sample_time = sampler.sample()
        whole_time += sample_time

        sampled_df = sampled_df[['Content', 'EventTemplate']]
        sampled_df = sampled_df.applymap(correct_message)
        sampled_df.to_csv(
            f"new_sample/{dataset}_sampled_examples.csv", index=False)

        print(f"{dataset} dataset sampled")

    print(f"Avg sampling time: {whole_time/len(datasets):.4f}s")
