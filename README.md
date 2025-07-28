# RAG in the Wild: Mixture-of-Knowledge Retrieval Augmentation Framework

This repository contains the official codebase for our paper:

**"RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge Retrieval Augmentation"**  

---

## Overview

This framework provides a modular and extensible pipeline for evaluating Large Language Models (LLMs) in Retrieval-Augmented Generation (RAG) settings, using diverse knowledge sources (“mixture-of-knowledge” retrieval) across a variety of QA benchmarks (MMLU, CSBench, ARC, OBQA, and more).  
It is designed for high-throughput, multi-domain experiments to rigorously analyze the benefits and limitations of RAG, as explored in our paper.

---

## Features

- **Plug-and-play with multiple LLMs** (e.g., Llama, Qwen, etc.), using [vLLM](https://github.com/vllm-project/vllm) for efficient inference.
- **Flexible prompt templating** for span extraction, multiple-choice, and true/false question types.
- **Mixture-of-knowledge retrieval:** Evaluate model performance with retrieved evidence from a diverse set of knowledge domains.
- **Structured outputs** for easy benchmarking and downstream analysis.
- **Easy extensibility:** Add new datasets, domains, or prompt styles with minimal code changes.

---

## Setup

**Requirements**
- Python 3.8+
- [transformers](https://github.com/huggingface/transformers)
- [vllm](https://github.com/vllm-project/vllm)
- [tqdm](https://github.com/tqdm/tqdm)

Install dependencies:
```bash
pip install transformers vllm tqdm
```
--- 

## Usage

### Data Preparation

All the data have been uploaded to the huggingface repo [ritaranx/RAG_in_the_Wild](https://huggingface.co/datasets/ritaranx/RAG_in_the_Wild).


Each task/domain file should be located at:

```retrieve/<dataset>/<domain>/<task>.json```

Each file should be a JSON list of examples with at least:

```[
  {
    "question": "...",
    "answer": "...",
    "ctxs": [...],  // List of passages (strings or [passage, ...] for some settings)
    "multichoice_options": [...] // (optional for MC tasks)
  },
  ...
]
```

## Running Inference
Basic usage:
```
python main.py \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --model_path /path/to/llama3.1-8b-instruct \
  --model_name llama-3.1-8b \
  --temperature 0.0 \
  --dataset csbench \
  --tensor_parallel_size 1
```

Main arguments:
- ```--tokenizer```: HuggingFace tokenizer name or path.
- ```--model_path```: Local path to vLLM-compatible model weights.
- ```--model_name```: Used for output folder naming.
- ```--temperature```: Decoding temperature.
- ```--top_p```: Nucleus sampling parameter (default 0.99).
- ```--tensor_parallel_size```: Number of GPUs.
- ```--dataset```: Dataset name (e.g., mmlu_dev, csbench, arc_c).

Results are saved to:
```inference/<dataset>/<domain>/<task>/prompts_rag_<model_name>/corpus_agg.txt```


## Citation
If you find this repo useful, please kindly cite the following paper:
```
@article{xu2025rag,
  title   = {RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge Retrieval Augmentation},
  author  = {Ran Xu and Yuchen Zhuang and Yue Yu and Haoyu Wang and Wenqi Shi and Carl Yang},
  journal = {arXiv preprint},
  year    = {2025},
}
```
