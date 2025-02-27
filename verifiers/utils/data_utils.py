import random
from typing import List, Dict, Optional, Callable

from datasets import Dataset, load_dataset  # type: ignore

def extract_boxed_answer(text: str) -> Optional[str]:
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    boxed_start = text.find('\\boxed{')
    if boxed_start == -1:
        return None
    content_start = boxed_start + 7
    closing_brace = find_matching_brace(text, content_start)
    return text[content_start:closing_brace] if closing_brace != -1 else None

def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_prompt(prompt: str,
                  system_prompt: Optional[str] = None,
                  few_shot: Optional[List[Dict[str, str]]] = None,
                  fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": prompt})
    return messages

def preprocess_dataset(
    dataset_name: str = "gsm8k",
    split: str = "train",
    system_prompt: Optional[str] = None,
    few_shot: Optional[List[Dict[str, str]]] = None,
    fewshot_prob: float = 1.0,
    prompt_formatter: Optional[Callable] = None,
    answer_extractor: Optional[Callable] = None
) -> Dataset:
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")[split]
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["question"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_hash_answer(x["answer"])
        })
        return dataset
    elif dataset_name == "math":
        dataset = load_dataset("chiayewken/competition_math")[split]
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["problem"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_boxed_answer(x["solution"])
        })
        return dataset
    elif dataset_name == "openbookqa":
        dataset: Dataset = load_dataset("allenai/openbookqa", "main")[split]

        def format_question(example):
            choices_texts = example['choices']['text']
            choices_labels = example['choices']['label']

            formatted_choices = []
            for i in range(len(choices_labels)):
                formatted_choices.append(f"{choices_labels[i]}. {choices_texts[i]}")

            question = f"Question: {example['question_stem']}\n\nChoices:\n" + "\n".join(formatted_choices)
            return question

        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(
                format_question(x),
                str(system_prompt) + "\n\nReturn only the letter of the correct answer.",
                few_shot,
                fewshot_prob
            ),
            "answer": x["answerKey"]
        })
        return dataset
    elif dataset_name == "quest-corruption":
        dataset = load_dataset("Quest-AI/quest-corruption-truncated4grpo-2k-dataset-v1", split=split)
        dataset = dataset.map(lambda x: {
            "prompt": [{
                "role": "user",
                "content": (
                    prompt_formatter(x)
                    if prompt_formatter
                    else x["corrupted"]
                )
            }],
            "answer": (
                answer_extractor(x)
                if answer_extractor
                else x["original"]
            )
        })
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
