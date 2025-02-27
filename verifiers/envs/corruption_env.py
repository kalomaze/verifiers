from typing import List, Dict, Any, Tuple, Union, Sequence
from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams
from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import CorruptionRubric
from verifiers.utils import preprocess_dataset

class CorruptionEnv(SimpleEnv):
    def __init__(self,
                 dataset: str = "quest-corruption",
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 fields: List[str | Tuple[str, ...]] = ["original"],
                 **kwargs):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot, **kwargs)
        self.parser = XMLParser(fields=fields)
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=system_prompt,
            few_shot=few_shot,
            prompt_formatter=self._format_prompt,
            answer_extractor=lambda x: x["original"]
        )
        self.eval_dataset = None
        self.rubric = CorruptionRubric()

    def _format_prompt(self, example: Dict) -> str:
        return (
            f"{example['corrupted']}\n\n"
            "<objective>\n"
            "gently repair the <original> content\n"
            "</objective>\n\n"
            "<original>\n"
        )

    def get_dataset(self, **kwargs: Any):
        return self.dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def eval(self, batch_size: int = 10, **kwargs: Any):
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="test",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot,
                prompt_formatter=self._format_prompt,
                answer_extractor=lambda x: x["original"]
            )

    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 output_type: str = "ids",
                 **kwargs: Any) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:
        return super().generate(
            prompts=prompts,
            llm=llm,
            sampling_params=sampling_params,
            output_type=output_type,
            use_chat=False,
            **kwargs
        )