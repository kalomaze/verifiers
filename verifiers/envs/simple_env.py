import json
import random
from typing import List, Dict, Any, Sequence, Union

from vllm import LLM, SamplingParams
from verifiers.envs.environment import Environment

class SimpleEnv(Environment):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)

    def format_prompt(self, prompt: str, fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.few_shot and random.random() < fewshot_prob:
            messages.extend(self.few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 output_type: str = "ids",
                 use_chat: bool = True,
                 **kwargs: Any) -> Union[Dict[str, Any], List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:

        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        states = []
        if use_chat:
            completions = llm.chat(prompts, sampling_params=custom_sp, use_tqdm=False)
            for i, completion in enumerate(completions):
                state = {
                    "messages": prompts[i] + [{"role": "assistant", "content": completion.outputs[0].text}],
                    "prompt_ids": list(completion.prompt_token_ids),
                    "completion_ids": list(completion.outputs[0].token_ids),
                    "completion_mask": [1] * len(completion.outputs[0].token_ids)
                }
                states.append(state)
        else:
            text_prompts = [p[-1]['content'] for p in prompts]
            outputs = llm.generate(text_prompts, sampling_params=custom_sp)
            for i, output in enumerate(outputs):
                state = {
                    "messages": [{"role": "user", "content": text_prompts[i]},
                                 {"role": "assistant", "content": output.outputs[0].text}],
                    "prompt": text_prompts[i],
                    "completion": output.outputs[0].text,
                    "prompt_ids": output.prompt_token_ids,
                    "completion_ids": output.outputs[0].token_ids,
                    "completion_mask": [1] * len(output.outputs[0].token_ids)
                }
                states.append(state)

        # Logging
        if states:
            self.logger.debug(f"Prompt 0 IDs: {states[0]['prompt_ids']}")
            self.logger.debug(f"Completion 0 IDs: {states[0]['completion_ids']}")
            if use_chat:
                self.logger.info(
                    "Prompt 0 Messages:\n%s\n\nCompletion 0:\n%s",
                    json.dumps(states[0]["messages"][:-1], indent=4),
                    json.dumps(states[0]["messages"][-1], indent=4)
                )
            else:
                self.logger.info(
                    "Prompt 0 Text:\n%s\n\nCompletion 0:\n%s",
                    states[0]["prompt"],
                    states[0]["completion"]
                )

        # Return formatted output
        if output_type == "ids":
            return {
                "ids": [s["completion_ids"] for s in states],
                "messages": [s["messages"][-1:] for s in states],
                "mask": [s["completion_mask"] for s in states]
            }
        elif output_type == "text":
            return [s["completion"] for s in states]
        elif output_type == "messages":
            return [s["messages"] for s in states]
        else:
            raise ValueError(f"Invalid output type: {output_type}")
