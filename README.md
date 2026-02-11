# VarParser: Unleashing the Neglected Power of Variables for LLM-based Log Parsing

This repository contains the official implementation of our WWW 2026 paper:

**"VarParser: Unleashing the Neglected Power of Variables for LLM-based Log Parsing".** 

*Proceedings of the ACM Web Conference (WWW), 2026*

## Clone the Repository

```bash
git clone https://github.com/mianmaner/VarParser.git
cd VarParser
```

## Repository Structure

```txt
📦 VarParser
├─ dataset/
│  └─ 2k_dataset/            # LogHub-2k datasets
│  └─ full_dataset/          # LogHub-2.0 datasets
├─ sample/                   # Sampled candidate logs
├─ parsed/                   # Parsed log results
├─ result/                   # Evaluation outputs
├─ varparser/                # Core implementation
│  ├─ cache.py               # Variable-aware template cache
│  ├─ prompt.py              # LLM prompt construction & querying
│  ├─ utils.py               # Log preprocessing utilities
│  └─ eval.py                # Parsing evaluation
├─ running.py                # Main entry for online log parsing
├─ sample.py                 # Candidate sampling script
├─ config.yaml               # LLM configuration
├─ requirements.txt
└─ README.md
```

## Prerequisites

Ensure you have Python 3.11+ installed.
```txt
pip install -r requirements.txt
```

Then set your OpenAI API information in `config.yaml`.

```txt
model: "<MODEL_NAME>"
api_key: "<OpenAI_API_KEY>"
base_url: "<OpenAI_BASE_URL>"
```

## Minimal Running Example
You can excute the following command to quickly run VarParser:

```bash
bash run_parser.sh Thunderbird 2k
```

This command performs online log parsing on the Thunderbird dataset using the 2k log subset.

By default, the script will parse all datasets in LogHub-2.0 using the specified LLM configuration.

Parsed log results will be saved to:
```txt
parsed/
```

Evaluation results (e.g., parsing accuracy, runtime, token usage) will be saved to:
```txt
result/
```

## Dataset

We conduct experiments on LogHub-2.0, a large-scale and widely used log parsing benchmark.

Please download the datasets from Zenodo:
🔗 https://zenodo.org/record/8275861

Unzip the files into the following directory:
```txt
dataset/full_dataset/
```

Example structure:

```txt
📦 VarParser
├─ dataset
|  └─ full_dataset
│     ├─ Apache
│     │  ├─ Apache_full.log
│     │  ├─ Apache_full.log_structured.csv
│     │  ├─ Apache_full.log_structured_corrected.csv
│     │  ├─ Apache_full.log_templates.csv
│     │  └─ Apache_full.log_templates_corrected.csv
│     ├─ ...
```

## Reducible Worflow

Ensure that you have downloaded the LogHub-2.0 dataset. Then you can excute the following command to start online parsing with VarParser.

```bash
bash run_parser.sh all full
```

By default, the script will parse all datasets in LogHub-2.0 using the specified LLM configuration.

Parsed log results will be saved to:
```txt
parsed/
```

Evaluation results (e.g., parsing accuracy, runtime, token usage) will be saved to:
```txt
result/
```


### Candidate Sampling (Optional)

We have provided the saved sampled candidate logs for reproducing. You can also delete the file in `sample/` directory and excute `python sample.py`

```bash
python sample.py
```

This step is optional but recommended when adapting VarParser to new datasets.