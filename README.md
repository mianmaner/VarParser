# VarParser: Unleashing the Neglected Power of Variables for LLM-based Log Parsing


### Datasets

Please download the large-scale datasets of Loghub-2.0 via [Zenodo](https://zenodo.org/record/8275861) and unzip these datasets into the directory of `dataset/full_dataset`.

### Environment

```txt
python >= 3.11,
pip install -r requirements.txt
```

### Reproducible workflow

Please first set your OpenAI API in `config.json`.

```txt
"api_key": "<OpenAI_API_KEY>",
"base_url": "<OpenAI_BASE_URL>"
```

Then you can excute the following command to start online parsing.

```bash
python main.py --model='MODEL_NAME'
```

The parsed results will be saved in the `parsed/` directory and evaluation results will be saved in the `result/` directory.

> Candidate Sampling (optional): We have provided the saved sampled candidate logs for reproducing. You can also delete the file in `sample/` directory and excute `python sample.py`

### Baselines

Here are the links to the implementation code of all baselines:
- [Drain](https://github.com/logpai/loghub-2.0/blob/main/benchmark/old_benchmark/Drain_benchmark.py), 
- [AEL](https://github.com/logpai/loghub-2.0/blob/main/benchmark/old_benchmark/AEL_benchmark.py), 
- [LogPPT](https://github.com/logpai/loghub-2.0/tree/main/benchmark/LogPPT), 
- [UniParser](https://github.com/logpai/loghub-2.0/tree/main/benchmark/UniParser), 
- [LibreLog](https://github.com/zeyang919/LibreLog), 
- [LILAC](https://github.com/logpai/LILAC), 
- [LogBatcher](https://github.com/LogIntelligence/LogBatcher), 
- [LUNAR](https://github.com/logpai/LUNAR)

