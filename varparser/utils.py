from Levenshtein import distance
from itertools import groupby
from nltk.corpus import wordnet
import argparse
import calendar
import string
import re


def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def levenshtein_similarity(str1, str2):
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1 - distance(str1, str2) / max_len


def remove_punctuation(message):
    return message.translate(str.maketrans('', '', string.punctuation))


weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
weekday_pt = r'(' + '|'.join(weekdays) + ')'
month_pt = r'(' + '|'.join(months) + ')'


def create_fuzzy_key(message):
    # Variable Fuzzing
    patterns = [
        r'/\S+',  # URL
        r'\w+@[\S]+',  # Email
        r'(\w\+|\w\-)',  # Direction
        r'(\w{2}:){5}\w{2}',  # MAC
        r'\S*\d\S*',  # Number
        r'\S+\.(com|net|\w{2})',  # Website
        weekday_pt,  # Weekday
        month_pt,  # Month
    ]

    for pattern in patterns:
        message = re.sub(pattern, "{Variable}", message)
    message = message.replace('{Variable} ', '{Variable}')
    message = re.sub(r'({Variable}){1,}', '{Variable}', message)

    # Basic preprocessing
    message = remove_punctuation(message)
    message = re.sub(r'\s+', ' ', message)
    message = re.sub(r'\S*\d\S*', '*', message)
    message = ' '.join([token for token, _ in groupby(message.split())])
    return message.strip()


def process_template(template):
    pattern = r'\{(\w+)\}'
    template_tokens = template.split()
    for index, token in enumerate(template_tokens):
        if re.search(pattern, token) or '<*>' in token:
            template_tokens[index] = '<*>'

    corrected_tokens = []
    for token in template_tokens:
        if '<*>' in token:
            if len(corrected_tokens) == 0 or corrected_tokens[-1] != '<*>':
                corrected_tokens.append('<*>')
        elif len(remove_punctuation(token)) != 0:
            corrected_tokens.append(token)

    new_template = " ".join(corrected_tokens)
    new_template = re.sub(r'\s+', ' ', new_template)
    new_template = new_template.strip(".\" ")
    return new_template


def _tokenize(log_content, tokenize_pattern=r'[ ,|\{\}]'):
    words = re.split(tokenize_pattern, log_content)
    new_words = []

    for word in words:
        if '=' in word:
            parts = word.split('=')
            if len(parts) <= 2:
                new_words.append(parts[0])
            # else:  # Assuming this is for handling URL parameters
                # pass
        elif '/' in word.lower():
            continue
        else:
            word = re.sub(r'\d+(\.\d+)?', '*', word)
            new_words.append(word)

    new_words = [word for word in new_words if word]  # Remove empty strings
    if not new_words:
        new_words.append(re.sub(r'\d+(\.\d+)?', '*', log_content))

    return new_words


def tokenize(log_messages):
    log_tokens_list = []
    for log_message in log_messages:
        log_message = correct_message(log_message)
        log_tokens = _tokenize(log_message)
        log_tokens_list.append(log_tokens)
    return log_tokens_list


def correct_message(message):
    message = re.sub(r'\.{3,}', ' ', message)
    message = re.sub(r'\s+', ' ', message)
    message = message.strip(". ")
    return message


def correct_template(template):
    patterns = [
        (r'\d+:\d+', '{time}'),
        (r'(0x|0)[a-fA-F0-9]+', '{hex_code}'),
        (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '{ip_address}'),
        (r'([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', '{mac_address}'),
        (r'[\S]+@[\S]+', '{email}'),
        (r'(=|==)(?!\s+\{)\S*', '{value}'),
    ]
    for pattern, replacement in patterns:
        template = re.sub(pattern, replacement, template)

    p_token = ""
    template_tokens = template.split()
    for index, token in enumerate(template_tokens):
        if (p_token == "=" or p_token == "==") or token == "root":
            template_tokens[index] = '{unknown}'
        elif re.match(r'\W*\d+\W*', token):
            template_tokens[index] = '{number}'
        elif re.match(r'\W*(file:|folder:)*\/+\S+(?:\/\S+)*\W*', token):
            template_tokens[index] = '{file_path}'
        p_token = token
    template = " ".join(template_tokens)
    return template
