from varparser.utils import correct_template, create_fuzzy_key, process_template, _tokenize, jaccard_similarity, levenshtein_similarity, remove_punctuation

from collections import defaultdict
from nltk.corpus import words
from tqdm import tqdm
import numpy as np
import pandas as pd
import calendar
import string
import os
import re


class TrieNode:
    def __init__(self):
        self.children = {}
        self.template_index = None


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.indexes = []

    def insert(self, template, index):
        if index in self.indexes:
            return
        self.indexes.append(index)
        node = self.root
        for token in _tokenize(template):
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.template_index = index

    def print(self, node=None, prefix=""):
        if node is None:
            node = self.root

        if node.template_index is not None:
            print(f"{prefix} (Index: {node.template_index})")

        for token, child in node.children.items():
            print(f"{prefix}-- {token}")
            self.print(child, prefix + "    ")

    def search(self, log_tokens):
        node = self.root
        i = 0  # Token index

        while i < len(log_tokens):
            token = log_tokens[i]
            if token in node.children:
                node = node.children[token]
                i += 1
            elif "<*>" in node.children:
                thres = 7
                if node == self.root:
                    thres = 2
                star_node = node.children["<*>"]
                begin = i
                # Enter the wildcard matching loop
                while i < len(log_tokens):
                    if log_tokens[i] in star_node.children:
                        node = star_node.children[log_tokens[i]]
                        i += 1
                        break
                    i += 1
                    if i - begin >= thres:
                        return self._collect_all_indices(node), False
                else:
                    if node.children['<*>'].template_index is not None:
                        return node.children['<*>'].template_index, True
                    else:
                        return self._collect_all_indices(node), False
            else:
                return self._collect_all_indices(node), False

        # Check if the node is a complete template
        if node.template_index is not None:
            return node.template_index, True
        else:
            return self._collect_all_indices(node), False

    def _collect_all_indices(self, node):
        indices = []
        if node.template_index is not None:
            indices.append(node.template_index)
        for child in node.children.values():
            indices.extend(self._collect_all_indices(child))
        return indices


class VarCache:
    def __init__(self, dataset, shot):
        self.dataset = dataset
        self.shot = shot
        self.common_words = words.words()
        self.dates = list(calendar.day_name) + list(calendar.day_abbr) + \
            list(calendar.month_name) + list(calendar.month_abbr)
        self.unlabel_cnt = 0

        self.var_units = defaultdict(dict)
        self.token_tree = Trie()
        self.matching_table = {}
        self.templates = []
        self.template_to_var = []
        self.template_examples = []
        self.template_tokens_list = []

    def extract_vars(self, sampled_df):
        sampled_df = sampled_df.head(self.shot)
        for index, row in sampled_df.iterrows():
            log_message = row['Content']
            template = row['EventTemplate']
            self.insert(log_message, template, update_token_tree=False)
        # self.token_tree.print()
        return

    def match(self, log_message):
        # self.token_tree.print()
        masked_msg = create_fuzzy_key(log_message)
        if masked_msg in self.matching_table:
            index = self.matching_table[masked_msg]
            return index, True

        log_tokens = _tokenize(log_message)
        if len(log_tokens) == 1 or set(log_tokens).issubset(set(self.common_words)):
            template = log_message
            if template in self.templates:
                index = self.templates.index(template)
            else:
                index = self.insert(log_message, template)
            return index, True

        result, success = self.token_tree.search(log_tokens)
        if success:
            index = result
            self.matching_table[masked_msg] = index
            self.update(log_message, index)
            return index, True

        for index in result:
            if self._match(log_message, index) and self.mask_similarity(log_tokens, index) >= 0.8:
                self.matching_table[masked_msg] = index
                self.update(log_message, index)
                return index, True
        return None, False

    def _match(self, log_message, index):
        log_tokens = _tokenize(log_message)
        template_tokens = self.template_tokens_list[index]
        example = self.template_examples[index][0]
        if log_tokens[0] == template_tokens[0]:
            return True
        if len(template_tokens) == 1:
            if template_tokens[0] == "<*>":
                msg = re.sub(r"\d", "*", log_message)
                exp = re.sub(r"\d", "*", example)
                if levenshtein_similarity(msg, exp) >= 0.8:
                    return True
        elif template_tokens[1] == log_tokens[1]:
            return True
        return False

    def update(self, log_message, index):
        log_tokens = log_message.split()
        template_tokens = self.templates[index].split()
        if len(log_tokens) != len(template_tokens):
            return
        for pos, log_token in enumerate(log_tokens):
            template_token = template_tokens[pos]
            if log_token != template_token:
                for label, pos_list in self.template_to_var[index].items():
                    if pos in pos_list:
                        self._insert(label, log_token)
                        break
                else:
                    label = f'unknown_{self.unlabel_cnt}'
                    self.unlabel_cnt += 1
                    self.template_to_var[index][label] = [pos]
                    self.var_units[label]['var'] = [
                        template_tokens[pos], log_token]
                    self.var_units[label]['symbol'] = [
                        template_tokens[pos], log_token]
                    if len(log_tokens) == len(self.templates[index].split()):
                        template_tokens[pos] = "<*>"
                        self.templates[index] = " ".join(template_tokens)
        return

    def insert(self, log_message, response, update_token_tree=True):
        response = self.insertion_process(log_message, response)
        response, label_vars, label_positions = self.updating_correction(
            log_message, response)

        template = correct_template(response)
        template = process_template(template)

        if template not in self.templates:
            template_tokens = self.post_process(template)
            for label, pos_list in label_positions.items():
                if label not in self.var_units:
                    self.var_units[label]['var'] = label_vars[label]
                    self.var_units[label]['symbol'] = [label_vars[label][0]]
                    self.var_units[label]['count'] = 1
                else:
                    for token in label_vars[label]:
                        self._insert(label, token)
            self.templates.append(template)
            self.template_examples.append([log_message, response])
            self.template_tokens_list.append(template_tokens)
            self.template_to_var.append(label_positions)
            result_index = len(self.templates) - 1
        else:
            for label, pos_list in label_positions.items():
                if label in self.var_units:
                    for token in label_vars[label]:
                        self._insert(label, token)
            result_index = self.templates.index(template)

        if update_token_tree and template != "<*>":
            self.token_tree.insert(template, result_index)
        masked_msg = create_fuzzy_key(log_message)
        self.matching_table[masked_msg] = result_index
        return result_index

    def _insert(self, label, token):
        if 'count' not in self.var_units[label]:
            self.var_units[label]['count'] = 1
        self.var_units[label]['count'] += 1
        var_list = self.var_units[label]['var']
        symbol_list = self.var_units[label]['symbol']
        if token.lower() not in var_list:
            var_list.append(token.lower())
        for symbol in symbol_list:
            if jaccard_similarity(token, symbol) >= 0.8:
                return
        symbol_list.append(token)
        return

    def adaptive_var_selection(self, log_message):
        log_tokens = _tokenize(log_message)
        special_tokens = []
        for token in log_tokens:
            if token not in self.common_words:
                special_tokens.append(token)

        # Sample relevant var_units
        max_sim = 0.5
        var_prompt = {}
        for token in special_tokens:
            max_score = 0
            temp_prompt = []
            for label, var_attr in self.var_units.items():
                if label.startswith('unknown'):
                    continue
                for symbol_var in var_attr['symbol']:
                    if len(symbol_var) == 0:
                        continue
                    mask_var = re.sub(r'\d', '*', symbol_var)
                    score = jaccard_similarity(token, mask_var)
                    if score > max_score:
                        max_score = score
                        temp_prompt = [label, symbol_var]
            # print(max_score)
            if max_score >= max_sim:
                var_prompt[temp_prompt[0]] = temp_prompt[1]
        # print(var_prompt)
        return var_prompt

    def get_reference(self, log_message):
        log_tokens = log_message.split()
        max_sim = -1
        reference = "env.createBean2(): Factory error creating {module} ({module}, {module})"
        for example in self.template_examples:
            template_tokens = example[0].split()
            if len(template_tokens) >= 30:
                continue
            sim = jaccard_similarity(log_tokens, template_tokens)
            if sim > max_sim:
                max_sim = sim
                reference = example[1]
        return reference

    def insertion_process(self, log_message, response):
        if log_message in response:
            response = log_message
        response = correct_template(response)
        flag = True
        for token in response.split():
            if not re.search(r'\{(.*?)\}', token):
                flag = False
                break
        if flag:
            res_tokens = response.split()
            start_token = log_message.split()[0]
            if not re.search(r'\d+', start_token):
                res_tokens[0] = start_token
            response = " ".join(res_tokens)
        return response

    def updating_correction(self, log_message, response):
        # Correction for llm hallucination
        label_vars, label_positions = self._extract_vars(log_message, response)
        if len(label_vars) == 0:
            return log_message, {}, {}
        temp_label_vars = label_vars.copy()
        for label, var_list in temp_label_vars.items():
            token = var_list[0]
            error = False
            if label.lower() == token.lower() or label.lower() == 'label':
                error = True
            if remove_punctuation(label) == '' or remove_punctuation(token) == '':
                error = True
            if self.is_common(token):
                error = True
            if len(token.split()) >= 9:
                error = True
            if error:
                response = response.replace(f"{{{label}}}", token)
                del label_vars[label]
                del label_positions[label]
        return response, label_vars, label_positions

    def mask_similarity(self, log_tokens, index):
        template_tokens = self.template_tokens_list[index]
        mask_poses = self.template_to_var[index]
        copied_log_tokens = log_tokens.copy()
        for label, pos_list in mask_poses.items():
            if label.startswith('unknown'):
                continue
            for pos in pos_list:
                if pos < len(copied_log_tokens):
                    copied_log_tokens[pos] = "<*>"
        # if template_tokens[-1] == "<*>":
        #     for idx in range(len(template_tokens), len(log_tokens)):
        #         copied_log_tokens[idx] = "<*>"
        return jaccard_similarity(copied_log_tokens, template_tokens)

    def is_common(self, token):
        token = remove_punctuation(token)
        if token.isupper():
            return False
        else:
            token = token.lower()
        if token in self.common_words:
            return True
        if token[-2:] == "ed" and token[:-2] in self.common_words:
            return True
        return False

    def post_process(self, template):
        template_tokens = _tokenize(template)
        for idx, token in enumerate(template_tokens):
            if '<*>' in token:
                template_tokens[idx] = '<*>'
        return template_tokens

    def update_var_count(self, index):
        for label in self.template_to_var[index].keys():
            if label in self.var_units:
                if 'count' not in self.var_units[label]:
                    self.var_units[label]['count'] = 1
                else:
                    self.var_units[label]['count'] += 1
        return

    def _extract_vars(self, log_message, template):
        pattern = r'\{(.*?)\}'
        labels = re.findall(pattern, template)

        if len(labels) > 30:
            return {}, {}

        label_count = {label: labels.count(label) for label in labels}
        template_regex = re.escape(template)
        for label in label_count:
            template_regex = template_regex.replace(
                r'\{' + re.escape(label) + r'\}', r'([\s\S]*)'
            )

        match = re.match(template_regex, log_message, re.DOTALL)
        if match:
            matched_vars = match.groups()

            var_start_positions = defaultdict(list)
            for i, var in enumerate(matched_vars, start=1):
                var_start_positions[var].append(match.start(i))

            label_vars = {label: [] for label in label_count}
            for index, label in enumerate(labels):
                label_vars[label].append(matched_vars[index])

            tokens = log_message.split()
            label_positions = defaultdict(list)
            token_start_positions = []
            token_end_positions = []

            current_position = 0
            for token in tokens:
                token_start_positions.append(current_position)
                current_position += len(token) + 1
                token_end_positions.append(current_position - 1)

            for label, value in zip(labels, matched_vars):
                start_poses = var_start_positions[value]
                for start_pos in start_poses:
                    end_pos = start_pos + len(value)
                    for idx, start in enumerate(token_start_positions):
                        end = token_end_positions[idx]
                        if (start_pos <= end and end_pos >= start) or (end_pos >= start and start_pos <= end):
                            if idx not in label_positions[label]:
                                label_positions[label].append(idx)

            return label_vars, label_positions
        else:
            print(f"Unmatch response: {log_message}")
        return {}, {}
