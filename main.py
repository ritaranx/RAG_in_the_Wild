import argparse
import copy
import json
import os
from tqdm import trange
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ------------- Prompt Templates ------------- #
PROMPTS = {
    "rag_span": """You have the following context passages:

Context: ${context}

Answer the following questions with one or a list of entities with the given context as the reference. Do not give a detailed explanation. Answer needs to be as short as possible.

Question: ${question}""",
    "qa_span": """
Answer the following questions with one or a list of entities. Do not give a detailed explanation. Answer needs to be as short as possible.

Question: ${question}
""",
    "qa_choice": """Given the answer candidates, please answer the following questions by selecting one choice.

Question: ${question}

Choice: ${choice}

Please answer with the capitalized alphabet only, without adding any extra phrase or period. Do not give a detailed explanation. 
""",
    "rag_choice": """You have the following context passages:

Context: ${context}

Please answer the following questions by selecting one choice, with the above context as the reference.

Question: ${question}

Choice: ${choice}

Please answer with the capitalized alphabet only, without adding any extra phrase or period. Do not give a detailed explanation. 
""",
    "rag_tf": """You have the following context passages:

Context: ${context}

Answer the following questions with True or False with the given context as the reference. Do not give a detailed explanation. 

Question: ${question}

Your answer should only be True or False.
""",
    "qa_tf": """Answer the following questions with True or False.

Question: ${question}

Your answer should only be True or False.
"""
}

TASK_DICT = {
    "arc_c": ["arc_c_test"],
    "simpleqa": ["simpleqa"],
    "csbench": ["csbench_assert", "csbench_mc"],
    "obqa": ["tqa"],
    "boolq": ["boolq", "strategyqa"],
    "mmlu_dev": ['mmlu'],
    "mmlu_pro_dev": ['mmlu_pro']
}

FORMAT_DICT_QA = {
    "simpleqa": PROMPTS["qa_span"],
    "csbench_assert": PROMPTS["qa_tf"],
    "csbench_mc": PROMPTS["qa_choice"],
    "obqa_test": PROMPTS["qa_choice"],
    "obqa_train": PROMPTS["qa_choice"],
    "arc_c_train": PROMPTS["qa_choice"],
    "arc_c_test": PROMPTS["qa_choice"],
    "sciq_train": PROMPTS["qa_span"],
    "sciq_test": PROMPTS["qa_span"],
    "nq": PROMPTS["qa_span"],
    "tqa": PROMPTS["qa_span"],
    "boolq": PROMPTS["qa_tf"],
    "strategyqa": PROMPTS["qa_tf"]
}
FORMAT_DICT_RAG = {
    "simpleqa": PROMPTS["rag_span"],
    "csbench_assert": PROMPTS["rag_tf"],
    "csbench_mc": PROMPTS["rag_choice"],
    "arc_c_test": PROMPTS["rag_choice"],
    "sciq_test": PROMPTS["rag_span"],
}
SPAN_TASKS = ["simpleqa", "sciq_test"]
TF_TASKS = ["csbench_assert", "boolq", "strategyqa"]

# ------------- Argument Parser ------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    return parser.parse_args()

# ------------- Helper Functions ------------- #
def format_choices(choices):
    labels = [chr(ord('A') + i) for i in range(len(choices))]
    return '\n'.join([f"{label}: {choice}" for label, choice in zip(labels, choices)])

def get_domains():
    return [
        "pubmed", "dpr_wiki", "pes2o_0.1", "rpj_c4_0.1", "rpj_github_0.1",
        "math_0.1", "rpj_stackexchange_0.1", "rpj_book_0.1", "rpj_arxiv_0.1",
        "rpj_common_crawl_2022_05_0.1",
        "pubmed_rerank", "dpr_wiki_rerank", "pes2o_0.1_rerank", "rpj_c4_0.1_rerank",
        "rpj_github_0.1_rerank", "math_0.1_rerank", "rpj_stackexchange_0.1_rerank",
        "rpj_book_0.1_rerank", "rpj_arxiv_0.1_rerank", "rpj_common_crawl_2022_05_0.1_rerank",
        "ensemble_rerank", "ensemble_retrieve"
    ]

def build_prompt(task, data, is_rag=True):
    context = "\n" + "".join(
        f"Passage {i+1}: {ctx}\n\n"
        for i, ctx in enumerate(data["passage"][:10][::-1])
    )
    if is_rag:
        fmt = FORMAT_DICT_RAG[task]
    else:
        fmt = FORMAT_DICT_QA[task]
    prompt = copy.deepcopy(fmt)
    prompt = prompt.replace("${question}", data["question"])
    prompt = prompt.replace("${context}", context)
    if "choices" in data:
        prompt = prompt.replace("${choice}", format_choices(data["choices"]))
    return [{"role": "user", "content": prompt.strip()}]

def load_examples(dataset, domain, task):
    path = f"retrieve/{dataset}/{domain}/{task}.json"
    with open(path, "r") as f:
        return json.load(f)

def extract_passages(example, domain):
    if domain in ["ensemble_rerank", "ensemble_retrieve"]:
        return [x[0] for x in example["ctxs"] if 13 < len(x[0])][:5]
    elif "rerank" in domain:
        return [x for x in example["rerank_ctxs"] if 13 < len(x)][:5]
    else:
        return [x for x in example["ctxs"] if 13 < len(x)][:5]

def main():
    args = parse_args()
    model_name, dataset = args.model_name, args.dataset

    # Prepare task list
    tasks = TASK_DICT.get(dataset, [
        x.split(".json")[0] for x in os.listdir(dataset)
    ])

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, repetition_penalty=1.05, max_tokens=512
    )
    llm = LLM(
        model=args.model_path,
        max_model_len=30000,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    domains = get_domains()

    print(f"Model: {args.model_name}, Tasks: {tasks}")

    for task in tasks:
        for domain in domains:
            output_path = f"inference/{dataset}/{domain}/{task}/prompts_rag_{model_name}/corpus_agg.txt"
            if os.path.exists(output_path):
                print(f"Path exists: {output_path}")
                continue
            print(f"Running task: {task} | domain: {domain}")

            # Load and preprocess examples
            examples = load_examples(dataset, domain, task)
            datas = []
            for ex in examples:
                data = {
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "passage": extract_passages(ex, domain)
                }
                if task not in SPAN_TASKS and task not in TF_TASKS:
                    data["choices"] = ex["multichoice_options"]
                datas.append(data)

            # Build prompts
            prompts = [build_prompt(task, data) for data in datas]

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            for i in trange(len(prompts)):
                text = tokenizer.apply_chat_template(
                    prompts[i],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                outputs = llm.generate([text], sampling_params)
                generated_text = outputs[0].outputs[0].text
                datas[i]["prediction"] = generated_text
                if i % 500 == 0:
                    print(f"[{i}] Sample: {datas[i]}")

            # Save predictions
            with open(output_path, "w") as f:
                for data in datas:
                    f.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    main()
